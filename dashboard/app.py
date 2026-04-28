"""
GateHub Guard Dashboard — Streamlit UI.

Run with:
    streamlit run dashboard/app.py
"""

import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import streamlit as st
import pandas as pd

from database.session import (
    init_db,
    get_active_sessions,
    vehicle_entry,
    vehicle_exit,
    is_blacklisted,
    add_to_blacklist,
    remove_from_blacklist,
    get_blacklist,
    OVERSTAY_MINUTES,
)

st.set_page_config(page_title='GateHub', layout='wide')
init_db()

GATES = ['Gate 1', 'Gate 2', 'Gate 3', 'Gate 4']

_DET_CANDIDATES = [
    'runs/detect/runs/detect/ablation_s-2/weights/best.pt',
    'runs/detect/runs/detect/ablation_s/weights/best.pt',
    'runs/detect/runs/detect/ablation_n-2/weights/best.pt',
    'runs/detect/runs/detect/ablation_n/weights/best.pt',
    'runs/detect/runs/detect/ablation_m/weights/best.pt',
    'runs/detect/gatehub/weights/best.pt',
]
_CRNN_WEIGHTS = 'checkpoints/crnn/best.pt'


# ── pipeline ──────────────────────────────────────────────────────────

@st.cache_resource(show_spinner='Loading GateHub models…')
def _load_pipeline():
    from pipeline.inference import GateHubPipeline
    det = next((p for p in _DET_CANDIDATES if Path(p).exists()), None)
    if det is None or not Path(_CRNN_WEIGHTS).exists():
        return None
    try:
        return GateHubPipeline(det, _CRNN_WEIGHTS)
    except Exception as exc:
        st.warning(f'Pipeline load failed: {exc}')
        return None


# ── image helpers ─────────────────────────────────────────────────────

def _annotate(frame: np.ndarray, results: list[dict]) -> np.ndarray:
    """Draw bounding boxes + labels on frame, return RGB numpy array."""
    out = frame.copy()
    for r in results:
        x1, y1, x2, y2 = r['bbox']
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 0), 2)
        label = f"{r['plate']}  {r['det_conf']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 220, 0), -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def _decode_upload(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


@st.cache_data(show_spinner=False, max_entries=3)
def _scan_video(_pipeline, video_bytes: bytes) -> tuple:
    """Scan video bytes for plates; return (annotated_rgb | None, results)."""
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        f.write(video_bytes)
        tmp = Path(f.name)
    try:
        cap = cv2.VideoCapture(str(tmp))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        step = max(1, total // 60)
        best_frame, best_results, best_conf = None, [], 0.0
        idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if idx % step == 0:
                res = _pipeline.process_frame(frame)
                if res:
                    conf = max(r['det_conf'] for r in res)
                    if conf > best_conf:
                        best_conf = conf
                        best_frame = frame.copy()
                        best_results = res
            idx += 1
        cap.release()
    finally:
        tmp.unlink(missing_ok=True)
    if best_frame is None:
        return None, []
    return _annotate(best_frame, best_results), best_results


# ── scan result display ───────────────────────────────────────────────

def _show_scan_result(annotated_rgb: np.ndarray, results: list[dict],
                      key_suffix: str) -> None:
    """Show annotated image and Use-for-Entry/Exit buttons."""
    st.image(annotated_rgb, use_container_width=True)
    if not results:
        st.warning('No plate detected. Try a clearer image or better lighting.')
        return
    best = max(results, key=lambda r: r['det_conf'])['plate']
    for r in results:
        st.success(f"Detected: **{r['plate']}**  (confidence {r['det_conf']:.2f})")
    st.divider()
    c1, c2 = st.columns(2)
    if c1.button(f'Use **{best}** for Entry',
                 use_container_width=True, key=f'use_entry_{key_suffix}'):
        st.session_state['scan_for_entry'] = best
        st.rerun()
    if c2.button(f'Use **{best}** for Exit',
                 use_container_width=True, key=f'use_exit_{key_suffix}'):
        st.session_state['scan_for_exit'] = best
        st.rerun()


# ── plate scanner widget ──────────────────────────────────────────────

def _scanner_widget(pipeline) -> None:
    if pipeline is None:
        st.info(
            'Models not loaded — scanner unavailable. '
            'Run `python -m detection.train` and `python -m ocr.train` first.'
        )
        return

    cam_tab, img_tab, vid_tab = st.tabs(['📷 Camera', '🖼️ Image Upload', '🎥 Video Upload'])

    with cam_tab:
        st.caption('Uses your browser camera. Point at the plate and click the capture button.')
        snap = st.camera_input('Capture plate', label_visibility='collapsed')
        if snap:
            frame = _decode_upload(snap.getvalue())
            results = pipeline.process_frame(frame)
            _show_scan_result(_annotate(frame, results), results, 'cam')

    with img_tab:
        upload = st.file_uploader('Upload image', type=['jpg', 'jpeg', 'png', 'bmp'],
                                  label_visibility='collapsed')
        if upload:
            frame = _decode_upload(upload.read())
            results = pipeline.process_frame(frame)
            _show_scan_result(_annotate(frame, results), results, 'img')

    with vid_tab:
        st.caption('Scans up to 60 evenly-spaced frames and returns the highest-confidence detection.')
        vid = st.file_uploader('Upload video', type=['mp4', 'avi', 'mov', 'mkv'],
                               label_visibility='collapsed')
        if vid:
            with st.spinner('Scanning video for plates…'):
                annotated_rgb, results = _scan_video(pipeline, vid.read())
            if annotated_rgb is not None:
                _show_scan_result(annotated_rgb, results, 'vid')
            else:
                st.warning('No plates detected in video.')


# ── tab helpers ───────────────────────────────────────────────────────

def _status_label(elapsed: float, plate: str, blacklisted_plates: set[str]) -> str:
    if plate in blacklisted_plates:
        return 'BLACKLISTED'
    if elapsed > OVERSTAY_MINUTES:
        return f'OVERSTAY ({elapsed:.0f} min)'
    return f'Active ({elapsed:.0f} min)'


# ── tabs ──────────────────────────────────────────────────────────────

def tab_sessions() -> None:
    st.header('Live Vehicle Sessions')

    if st.button('Refresh', key='refresh_sessions'):
        st.rerun()

    sessions = get_active_sessions()
    if not sessions:
        st.info('No vehicles currently on campus.')
        return

    blacklisted = {r['plate'] for r in get_blacklist()}
    rows, alerts = [], []

    for s in sessions:
        elapsed = float(s.get('elapsed_min') or 0)
        plate   = s['plate']
        label   = _status_label(elapsed, plate, blacklisted)
        rows.append({
            'Plate':         plate,
            'CNIC':          s.get('cnic') or '—',
            'Entry Gate':    s['entry_gate'],
            'Entry Time':    (s['entry_time'] or '')[:16],
            'Elapsed (min)': f'{elapsed:.0f}',
            'Status':        label,
        })
        if 'BLACKLISTED' in label or 'OVERSTAY' in label:
            alerts.append((plate, label))

    if alerts:
        st.subheader('Alerts')
        for plate, label in alerts:
            color = 'error' if 'BLACKLISTED' in label else 'warning'
            getattr(st, color)(f'{plate} — {label}')
        st.divider()

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def tab_entry_exit(pipeline) -> None:
    st.header('Entry / Exit')

    with st.expander('🔍 Scan Plate', expanded=True):
        _scanner_widget(pipeline)

    st.divider()

    # Inject scanned plate into the text_input widgets before they render
    if 'scan_for_entry' in st.session_state:
        st.session_state['in_plate'] = st.session_state.pop('scan_for_entry')
    if 'scan_for_exit' in st.session_state:
        st.session_state['out_plate'] = st.session_state.pop('scan_for_exit')

    col_entry, col_exit = st.columns(2)

    with col_entry:
        st.subheader('Log Entry')
        plate_in = st.text_input('Plate Number', key='in_plate').strip().upper()
        gate_in  = st.selectbox('Entry Gate', GATES, key='in_gate')
        cnic_in  = st.text_input('CNIC Number (optional)', key='in_cnic').strip() or None

        if st.button('Log Entry', type='primary', key='btn_entry'):
            if not plate_in:
                st.warning('Enter a plate number.')
            elif is_blacklisted(plate_in):
                st.error(f'ACCESS DENIED — {plate_in} is on the blacklist.')
            else:
                vehicle_entry(plate_in, gate_in, cnic_in)
                st.success(f'Entry logged: {plate_in} at {gate_in}.')

    with col_exit:
        st.subheader('Log Exit')
        plate_out = st.text_input('Plate Number', key='out_plate').strip().upper()
        gate_out  = st.selectbox('Exit Gate', GATES, key='out_gate')

        if st.button('Log Exit', type='primary', key='btn_exit'):
            if not plate_out:
                st.warning('Enter a plate number.')
            else:
                result = vehicle_exit(plate_out, gate_out)
                if result:
                    st.success(f'Exit logged: {plate_out} from {gate_out}.')
                else:
                    st.warning(f'No active session found for {plate_out}.')


def tab_blacklist() -> None:
    st.header('Blacklist Management')

    col_add, col_list = st.columns([1, 2])

    with col_add:
        st.subheader('Add Plate')
        bl_plate  = st.text_input('Plate Number', key='bl_plate').strip().upper()
        bl_reason = st.text_input('Reason',       key='bl_reason').strip()
        if st.button('Add to Blacklist', type='primary'):
            if bl_plate:
                add_to_blacklist(bl_plate, bl_reason)
                st.success(f'{bl_plate} added.')
                st.rerun()
            else:
                st.warning('Enter a plate.')

    with col_list:
        st.subheader('Current Blacklist')
        entries = get_blacklist()
        if not entries:
            st.info('Blacklist is empty.')
        else:
            for entry in entries:
                c1, c2, c3 = st.columns([2, 3, 1])
                c1.write(f"**{entry['plate']}**")
                c2.write(entry.get('reason') or '—')
                if c3.button('Remove', key=f"rm_{entry['plate']}"):
                    remove_from_blacklist(entry['plate'])
                    st.rerun()


# ── main ──────────────────────────────────────────────────────────────

def main() -> None:
    st.title('GateHub — Intelligent Vehicle Access Management')
    st.caption('NUST Campus | Deep Learning End-Semester Project')

    pipeline = _load_pipeline()

    t1, t2, t3 = st.tabs(['Live Sessions', 'Entry / Exit', 'Blacklist'])
    with t1:
        tab_sessions()
    with t2:
        tab_entry_exit(pipeline)
    with t3:
        tab_blacklist()


if __name__ == '__main__':
    main()
