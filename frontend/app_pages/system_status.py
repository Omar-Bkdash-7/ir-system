"""
System Monitor page — live health grid for all 7 services.
"""
from __future__ import annotations

import time

import streamlit as st

import config
import api.health as health_api
import api.retrieval as retrieval_api
from constants import (
    SS_AUTO_REFRESH,
    SS_AUTO_REFRESH_INTERVAL,
    SS_LAST_REFRESH_TIME,
    SS_RETRIEVAL_DEEP_STATUS,
    SS_SERVICE_HEALTH,
    SS_TEST_PING_RESULTS,
)


def _health_card(service_name: str, result: dict, test_key: str) -> None:
    ok = result.get("ok")
    ms = result.get("elapsed_ms", 0.0)
    checked = result.get("checked_at", "—")
    error = result.get("error", "")

    dot_class = "ir-dot--green" if ok else "ir-dot--red"
    status_text = "Healthy" if ok else "Down"
    border_color = "var(--ctp-green)" if ok else "var(--ctp-red)"

    with st.container(border=True):
        st.markdown(
            f'<div style="border-left:3px solid {border_color};padding-left:10px">'
            f'<div class="ir-badge">'
            f'<span class="ir-dot {dot_class}"></span>'
            f'<strong style="font-size:1em">{service_name}</strong>'
            f'</div>'
            f'<div style="color:var(--ctp-subtext0);font-size:0.82em;margin-top:4px">'
            f'{status_text} · {ms:.0f} ms · checked {checked}</div>'
            f'{"<div style=color:var(--ctp-red);font-size:0.8em>" + error + "</div>" if error and not ok else ""}'
            f'</div>',
            unsafe_allow_html=True,
        )
        info = config.SERVICES.get(service_name, {})
        st.caption(f":{info.get('port','?')} — {info.get('url','')}")

        test_result = st.session_state.get(SS_TEST_PING_RESULTS, {}).get(service_name)
        if test_result:
            col = "var(--ctp-green)" if test_result["ok"] else "var(--ctp-red)"
            st.markdown(
                f'<div style="color:{col};font-size:0.8em">Last manual ping: '
                f'{test_result["elapsed_ms"]:.0f} ms</div>',
                unsafe_allow_html=True,
            )

        if st.button("🔔 Test", key=test_key, use_container_width=True):
            pings = st.session_state.get(SS_TEST_PING_RESULTS, {})
            pings[service_name] = health_api.check_one_service(service_name)
            st.session_state[SS_TEST_PING_RESULTS] = pings
            st.rerun()


def render() -> None:
    st.markdown("## ⚙️ System Monitor")

    # ── Top controls ───────────────────────────────────────────────────────
    ctrl_c1, ctrl_c2, ctrl_c3 = st.columns([2, 2, 3])
    with ctrl_c1:
        if st.button("🔄 Refresh All", type="primary", use_container_width=True,
                     key="sysmon_refresh"):
            health_api.clear_cache()
            st.session_state[SS_LAST_REFRESH_TIME] = time.time()
            st.rerun()
    with ctrl_c2:
        auto = st.toggle(
            "Auto-refresh",
            value=st.session_state.get(SS_AUTO_REFRESH, False),
            key="sysmon_auto",
        )
        st.session_state[SS_AUTO_REFRESH] = auto
    with ctrl_c3:
        if auto:
            interval = st.selectbox(
                "Interval",
                config.AUTO_REFRESH_INTERVAL_OPTIONS,
                index=config.AUTO_REFRESH_INTERVAL_OPTIONS.index(
                    st.session_state.get(SS_AUTO_REFRESH_INTERVAL, 10)
                ),
                key="sysmon_interval",
            )
            st.session_state[SS_AUTO_REFRESH_INTERVAL] = interval
            last = st.session_state.get(SS_LAST_REFRESH_TIME)
            if last is None or (time.time() - last) >= interval:
                health_api.clear_cache()
                st.session_state[SS_LAST_REFRESH_TIME] = time.time()
                st.rerun()

    last_refresh = st.session_state.get(SS_LAST_REFRESH_TIME)
    if last_refresh:
        st.caption(f"Last refresh: {time.strftime('%H:%M:%S', time.localtime(last_refresh))}")

    st.divider()

    # ── Service health grid ────────────────────────────────────────────────
    st.markdown("### Services")
    health_data = health_api.check_all_services()

    svc_names = list(health_data.keys())
    cols_row1 = st.columns(4)
    cols_row2 = st.columns(3)
    all_cols = cols_row1 + cols_row2

    for col, svc_name in zip(all_cols, svc_names):
        with col:
            _health_card(
                svc_name,
                health_data[svc_name],
                test_key=f"sysmon_test_{svc_name.lower().replace(' ', '_')}",
            )

    # Summary badge
    n_ok = sum(1 for r in health_data.values() if r.get("ok"))
    n_total = len(health_data)
    if n_ok == n_total:
        st.success(f"✅ All {n_total} services are healthy.")
    else:
        st.error(f"❌ {n_total - n_ok} of {n_total} services are down.")

    st.divider()

    # ── Retrieval service deep status ──────────────────────────────────────
    st.markdown("### Retrieval Service — Index Readiness")
    if st.button("🔍 Check retrieval status", key="sysmon_retrieval_deep"):
        resp = retrieval_api.status()
        st.session_state[SS_RETRIEVAL_DEEP_STATUS] = resp.data if resp.ok else {"error": resp.error}
        st.rerun()

    deep = st.session_state.get(SS_RETRIEVAL_DEEP_STATUS)
    if deep:
        if "error" in deep:
            st.error(f"Retrieval service error: {deep['error']}")
        else:
            models_status = deep.get("models", {})
            dcols = st.columns(len(models_status) or 1)
            for col, (model, status) in zip(dcols, models_status.items()):
                with col:
                    color = "var(--ctp-green)" if status == "ready" else "var(--ctp-red)"
                    st.markdown(
                        f'<div class="ir-badge">'
                        f'<span class="ir-dot" style="background:{color}"></span>'
                        f'<strong>{model.upper()}</strong> — {status}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

    st.divider()

    # ── System info ────────────────────────────────────────────────────────
    st.markdown("### Configuration")
    import sys
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Environment", config.ENVIRONMENT)
        st.metric("Python", sys.version.split()[0])
    with c2:
        import streamlit as _st
        st.metric("Streamlit", _st.__version__)

    st.markdown("**Service URLs:**")
    url_rows = [{"Service": k, "URL": v["url"], "Port": v["port"]}
                for k, v in config.SERVICES.items()]
    import pandas as pd
    st.dataframe(pd.DataFrame(url_rows).set_index("Service"), use_container_width=True)
