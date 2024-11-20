from streamlit_folium import st_folium
from rank_bm25 import BM25Okapi
from kiwipiepy import Kiwi
import streamlit as st
import pandas as pd
import numpy as np
import folium
import time

st.set_page_config(layout="wide")

# 해시태그 목록
hashtag_list = [
    "좌석이 편해요", "반려동물과 가기 좋아요", "특별한 메뉴가 있어요", "향신료가 강하지 않아요", "매장이 청결해요",
    "화장실이 깨끗해요", "고기 질이 좋아요", "디저트가 맛있어요", "혼술하기 좋아요", "룸이 잘 되어있어요",
    "단체모임 하기 좋아요", "재료가 신선해요", "환기가 잘 돼요", "야외공간이 멋져요", "비싼 만큼 가치있어요",
    "뷰가 좋아요", "파티하기 좋아요", "인테리어가 멋져요", "아늑해요", "현지 맛에 가까워요", "포장이 깔끔해요",
    "빵이 맛있어요", "음악이 좋아요", "음료가 맛있어요", "잡내가 적어요", "아이와 가기 좋아요", "술이 다양해요",
    "가성비가 좋아요", "오래 머무르기 좋아요", "직접 잘 구워줘요", "음식이 맛있어요", "매장이 넓어요",
    "혼밥하기 좋아요", "라이브공연이 훌륭해요", "양이 많아요", "메뉴 구성이 알차요", "특별한 날 가기 좋아요",
    "음식이 빨리 나와요", "사진이 잘 나와요", "코스요리가 알차요", "차분한 분위기에요", "컨셉이 독특해요",
    "커피가 맛있어요", "샐러드바가 잘 되어있어요", "친절해요", "주차하기 편해요", "선물하기 좋아요",
    "집중하기 좋아요", "종류가 다양해요", "건강한 맛이에요", "반찬이 잘 나와요", "대화하기 좋아요", "기본 안주가 좋아요"
]

# 영-한 요일 딕셔너리
days_korean = {"Mon": "월요일", "Tue": "화요일", "Wed": "수요일", "Thu": "목요일", "Fri": "금요일",
               "Sat": "토요일","Sun": "일요일"}

# 데이터 로드
@st.cache_data
def load_data():
    time.sleep(0.5)
    filter_df = pd.read_csv("./recommend_system_data/filter_data.csv", encoding="utf-8")
    review_df = pd.read_csv("./recommend_system_data/combined_review_data_token(명사, 형용사).csv", encoding="utf-8")
    store_df = pd.read_csv("./recommend_system_data/store_data.csv", encoding="utf-8")

    review_df["tokens"] = review_df["tokens"].apply(lambda x: eval(x))
    return filter_df, review_df, store_df

# 사용자로부터 입력 받기
def get_user_input(filter_df):
    col1, col2 = st.columns([1, 2])
    with col1:
        category1 = st.selectbox("음식 종류 선택", ["전체"] + list(filter_df["category1"].unique()))
    with col2:
        hashtag = st.multiselect("해시태그 선택", hashtag_list)

    return category1, hashtag

def recommend_restaurants(user_input, review_df, store_df):
    kiwi = Kiwi()

    user_tokens = [
        token[0] for token in kiwi.tokenize(user_input) if token[1] in ["NNP", "NNG", "VA"]
    ] # 명사, 형용사만 토큰화

    # BM25 모델 생성
    bm25 = BM25Okapi(review_df["tokens"])
    bm25_scores = bm25.get_scores(user_tokens)

    review_data = pd.DataFrame({
        "name": review_df["name"],
        "bm25_score": bm25_scores
    })

    store_weights = (store_df.set_index("store_name")["weight"].reindex(review_data["name"]).fillna(0))
    review_data["adjusted_score"] = (review_data["bm25_score"] * 0.9 + store_weights.values * 0.1)

    top5 = review_data.nlargest(5, "adjusted_score")
    return top5[["name", "bm25_score"]]

# 지도 그리기
def display_map(store_info):
    latitude = store_info["latitude"]  # 위도
    longitude = store_info["longitude"]  # 경도

    store_map = folium.Map(
        location=[latitude, longitude],
        zoom_start=16,
        control_scale=False  # 스케일 표시 제거
    )

    folium.Marker(
        [latitude, longitude],
        icon=folium.Icon(icon="glyphicon-cutlery", color="red")
    ).add_to(store_map)

    st_folium(store_map, width=500, height=400)

# 추천된 식당을 화면에 표시
def display_recommendations(top_recommendations, store_df):
        st.markdown("<div style='margin-top:0; text-align: center; font-size: 30px;'><b>🍽 추천된 식당 리스트 🍽</b></div>", unsafe_allow_html=True)

        _, col1, _ = st.columns([1.5, 2, 1.5])
        with col1:
            selected_store = st.selectbox("목록에서 추천된 식당을 선택하면 식당과 셰프의 정보를 얻을 수 있습니다.", top_recommendations["name"])
            store_info = store_df[store_df["store_name"] == selected_store].iloc[0]

        col1, _, col2 = st.columns([1, 0.05, 1])
        if selected_store:
            with col1:
                st.markdown(
                    f"""
                    <div style="text-align: center; font-size: 25px;">
                        <b>출연진(셰프) 정보</b>
                    </div>""", unsafe_allow_html=True)

                if pd.notna(store_info["img_url"]):
                    st.markdown(f"""
                                <div style="text-align: center;">
                                    <img src="{store_info['img_url']}" alt="Store Image" style="height: 380px;">
                                </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                                <div style="text-align: center;">
                                    <img src="https://blog.kakaocdn.net/dn/dN7cN9/btsJEi4QvCB/bx1VvEzxeFPMF11sX1Stvk/img.png" alt="Store Image" style="height: 380px;">
                                </div>""", unsafe_allow_html=True)

                st.write("")

                if pd.notna(store_info['nickname']):
                    table_data = pd.DataFrame({
                        '출연자(별명)': [f"{store_info['cast']}({store_info['nickname']})"],
                        '수저': [store_info['spoon']],
                        '라운드': [store_info['round']]
                    })
                else:
                    table_data = pd.DataFrame({
                        '출연자': [store_info['cast']],
                        '수저': [store_info['spoon']],
                        '라운드': [store_info['round']]
                    })

                table_html = f"""
                <div style="display: flex; justify-content: center;">
                    <table style="width: 50%; margin: auto; border-collapse: collapse; text-align: center;" border="1">
                        <thead>
                            <tr>
                                {"".join([f'<th style="padding: 8px; text-align: center;">{col}</th>' for col in table_data.columns])}
                            </tr>
                        </thead>
                        <tbody>
                            {"".join([
                    "<tr>" + "".join([f'<td style="padding: 8px; text-align: center;">{cell}</td>' for cell in row]) + "</tr>"
                    for row in table_data.values
                ])}
                        </tbody>
                    </table>
                </div>
                """
                st.markdown(table_html, unsafe_allow_html=True)

            with col2:
                st.markdown(
                    f"""
                    <div style="text-align: center; font-size: 25px;">
                        <b>식당 정보</b>
                    </div>""", unsafe_allow_html=True)

                # 식당 정보 테이블 데이터 생성
                table_data = {
                    '항목': ['주소', '주차'],
                    '정보': [store_info.get('road_address', '정보 없음'), store_info.get('parking', '정보 없음')]
                }

                # 미슐랭 정보 추가
                if pd.notna(store_info.get('michelin')) and store_info['michelin']:
                    table_data['항목'].append('미슐랭')
                    table_data['정보'].append(store_info['michelin'])

                # 예약 링크 추가 (링크가 하이퍼링크로 작동하게 설정)
                reservation_links = []
                if str(store_info.get('naver_reservation')) == 'True':
                    reservation_links.append(f"<a href='{store_info['naver_url']}' target='_blank'>네이버 예약</a>")
                if str(store_info.get('catch_table_reservation')) == 'True':
                    reservation_links.append(f"<a href='{store_info['catch_table_url']}' target='_blank'>캐치테이블 예약</a>")

                # 예약 정보가 있을 경우 표에 추가
                if reservation_links:
                    table_data['항목'].append('예약 가능 사이트')
                    table_data['정보'].append(" / ".join(reservation_links))

                # HTML로 표 출력
                table_html = f"""
                <div style="display: flex; justify-content: center;">
                    <table style="width: 70%; margin: auto; border-collapse: collapse;" border="1">
                        <tbody>
                            {"".join([
                    f"<tr><td style='padding: 8px; font-weight: bold; width: 30%;'>{item}</td><td style='padding: 8px; width: 70%;'>{info}</td></tr>"
                    for item, info in zip(table_data['항목'], table_data['정보'])
                ])}
                        </tbody>
                    </table>
                </div>
                """
                st.markdown(table_html, unsafe_allow_html=True)  # 표 출력

                st.write("")

                _, col1, _ = st.columns([0.5, 2, 0.5])
                with col1:  # 가운데 열에 expander 배치
                    with st.expander("운영시간"):
                        for day in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]:
                            hours = store_info[day]
                            if hours:
                                if "(" in hours:  # 격주 운영 체크
                                    biweekly_info = hours.split("(")[1].strip(')')  # 괄호 안의 정보 추출
                                    st.write(
                                        f"{days_korean[day]}: {hours.split('(')[0].strip()} ({biweekly_info} 주에만 운영)")
                                else:
                                    st.write(f"{days_korean[day]}: {hours}")

                _, col1, _ = st.columns([0.5, 2, 0.5])
                with col1:
                    display_map(store_info)

if __name__ == "__main__":
    filter_df, review_df, store_df = load_data()  # 데이터 로드

    st.markdown("<h1 style='text-align: center; font-weight: bold;'>🍽 흑백요리사 식당 추천 🍽</h1>", unsafe_allow_html=True)
    _, col1, col2, _ = st.columns([0.2, 0.7, 1.5, 0.2])

    with col1:
        st.image("./recommend_system_data/system_image.png", width=380)

    with col2:
        ## ============== 카테고리, 해시태그 선택과 1차 데이터 필터링 ==============
        st.markdown("<div style='height: 80px;'></div>", unsafe_allow_html=True)  # 높이 50px의 빈 공간 추가

        with st.form("first_form"):
            # 필터와 해시태그 선택 및 1차 필터링
            category1, hashtag = get_user_input(filter_df)
            filtered_store_df = filter_df[filter_df["category1"] == category1].copy()  # 명시적으로 복사본 생성

            if category1 == "전체":
                filtered_store_df = filter_df.copy()  # 명시적으로 복사본 생성
            else:
                filtered_store_df = filter_df[filter_df["category1"] == category1].copy()  # 명시적으로 복사본 생성

            if hashtag:
                filtered_store_df.loc[:, "hashtag"] = filtered_store_df["hashtag"].replace(np.nan, "") # NaN을 빈 문자열로 대체
                filtered_store_df = filtered_store_df[filtered_store_df["hashtag"].apply(lambda x: isinstance(x, str) and any(tag in x for tag in hashtag))]

            filtered_store = filtered_store_df["store_name"].unique()
            st.write("")  # 간격 조절을 위한 빈줄 추가

            ## ============== 사용자 음식 취향 입력받기와 리뷰의 유사도 검정 후 식당 추천 ===============
            st.markdown("<p style='font-size:14px'>❗식당 취향을 입력해주세요 (예: 가족들과 함께 가면 좋은 분위기 좋은 레스토랑)</p>", unsafe_allow_html=True)
            col1, col2 = st.columns([6, 1])
            with col1:
                user_input = st.text_input("사용자 음식 선호 입력", "", label_visibility="collapsed") # 사용자 입력란에 빈 label을 주고 숨기기
            with col2:
                if st.form_submit_button("추천받기"):
                    if not review_df.empty and not filtered_store_df.empty:
                        filtered_review_df = review_df[review_df["name"].isin(filtered_store)]
                        recommended_stores = recommend_restaurants(user_input, filtered_review_df, store_df)

                        st.session_state.top_recommend_store = recommended_stores
                        st.session_state.store_df = store_df
                    else:
                        st.write("선택한 필터에 맞는 가게가 없습니다.")
    st.divider()

    # 추천된 식당, 셰프 정보 표시
    if "top_recommend_store" in st.session_state:
        display_recommendations(st.session_state.top_recommend_store, st.session_state.store_df)
