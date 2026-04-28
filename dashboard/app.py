"""
GateHub Guard Dashboard — Streamlit UI.

Run with:
    streamlit run dashboard/app.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

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


# ── helpers ───────────────────────────────────────────────────────────

def _status_label(elapsed: float, plate: str, blacklisted_plates: set[str]) -> str:
    if plate in blacklisted_plates:
        return 'BLACKLISTED'
    if elapsed > OVERSTAY_MINUTES:
        return f'OVERSTAY ({elapsed:.0f} min)'
    return f'Active ({elapsed:.0f} min)'


def _status_color(label: str) -> str:
    if 'BLACKLISTED' in label:
        return 'background-color: #ffcccc'
    if 'OVERSTAY' in label:
        return 'background-color: #fff3cc'
    return ''


# ── tabs ──────────────────────────────────────────────────────────────

def tab_sessions() -> None:
    st.header('Live Vehicle Sessions')

    sessions = get_active_sessions()
    if st.button('Refresh', key='refresh_sessions'):
        st.rerun()

    if not sessions:
        st.info('No vehicles currently on campus.')
        return

    blacklisted = {r['plate'] for r in get_blacklist()}
    rows = []
    alerts = []

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

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


def tab_entry_exit() -> None:
    st.header('Manual Entry / Exit')
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

    t1, t2, t3 = st.tabs(['Live Sessions', 'Entry / Exit', 'Blacklist'])
    with t1:
        tab_sessions()
    with t2:
        tab_entry_exit()
    with t3:
        tab_blacklist()


if __name__ == '__main__':
    main()
