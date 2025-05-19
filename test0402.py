import streamlit as st

st.title("위험 요소 체크")

# 위험 요소 리스트
hazards = [
    {"label": "냉정하지 위험", "disabled": False},
    {"label": "금접사", "disabled": False},
    {"label": "미끄럼 주의", "disabled": False},
    {"label": "혜청 차단", "disabled": True},
    {"label": "야생동물 출현", "disabled": True},
    {"label": "등산로 통제 어려움", "disabled": False},
]

selected = []

# 버튼 스타일로 표시 (streamlit badge 활용)
st.markdown("### 위험 요소 선택")
cols = st.columns(3)
for i, hazard in enumerate(hazards):
    with cols[i % 3]:
        if hazard["disabled"]:
            st.markdown(
                    f"<span style='background-color: lightgray; color: black; padding: 4px 8px; border-radius: 10px;'>{hazard['label']}</span>",
                    unsafe_allow_html=True
                )
            # st.badge(hazard["label"], variant="gray")
        else:
            if st.toggle(hazard["label"], key=hazard["label"]):
                selected.append(hazard["label"])

if st.button("Send"):
    if selected:
        st.success("선택된 항목: " + ", ".join(selected))
    else:
        st.info("선택된 항목이 없습니다.")
