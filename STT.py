import os
import glob
import json
from moviepy import VideoFileClip
import time
import pandas as pd
import streamlit as st
from st_aggrid import AgGrid

from faster_whisper import WhisperModel
import torch
import subprocess
torch.classes.__path__ = []

def load_videos(path='./data/'):

    try:
        vid_files = glob.glob(f'{path}*/*.mp4')


        vids = []
        for vid_file in vid_files:
            vid_meta = vid_file.replace('\\', '/').rsplit('/', 2)[-2:]

            vid_class = vid_meta[0]

            vid_no, vid_name = vid_meta[1].split('-')

            aud_name = False if not os.path.exists(vid_file.replace('.mp4', '.mp3')) else True # vid_file.replace('.mp4', '.mp3')
            scr_name = False if not os.path.exists(vid_file.replace('.mp4', '.csv')) else True # vid_file.replace('.mp4', '.csv')
            meta_name = False if not os.path.exists(vid_file.replace('.mp4', '.json')) else True # vid_file.replace('.mp4', '.json')

            seg_dir = False if not os.path.exists(vid_file.replace('.mp4', '/').replace('tutorials', 'segmented/')) else True



            vids.append([vid_class, vid_no, vid_name, vid_file, aud_name, scr_name, meta_name, seg_dir])
        


        df = pd.DataFrame(vids, columns=['Class', 'No', 'Name', 'Path', 'Audio', 'Script', 'Meta', 'Segmented']) # 
        return df
    except Exception as e:
        st.error(f"Error loading videos: {e}")
        return None

def load_seg_videos(path='./data/'):
    vid_files = glob.glob(f'{path}\*.mp4')
    vids = []
    for vid_file in vid_files:
        vid_name = vid_file.rsplit('\\', 1)[-1]
        vid_meta = vid_name.split('_')
        
        vids.append([vid_meta[0], vid_meta[1], vid_meta[2], vid_meta[3], vid_file])
    
    if len(vids) == 0:
        return None 
        
    df = pd.DataFrame(vids, columns=['Segment', 'Start', 'End', 'Name', 'Path'])
    return df
    
    
def split_video_by_seconds(video_path, output_dir, time_ranges):

    video = VideoFileClip(video_path)

    output_dir = video_path.replace('.mp4', '/').replace('tutorials', 'segmented/')
    

    os.makedirs(output_dir, exist_ok=True)
    
    for idx, (start, end, topic) in enumerate(time_ranges):
        clip = video.subclipped(start, end)
        output_filename = os.path.join(output_dir, f"s{str(idx+1).zfill(3)}_{int(start)}_{int(end)}_{topic}.mp4")
        clip.write_videofile(output_filename, codec="libx264")


@st.cache_resource
def load_whisper_model():
    model = WhisperModel("medium", device= "cuda", compute_type="float32")
    return model


if __name__ == '__main__':
    st.set_page_config(layout="wide")
    st.header("2025년 AI기반 농업교육포털 기능고도화 사업")

    w_model = load_whisper_model()
    
    df = load_videos(path='./data/tutorials/')


    tutorials = AgGrid(df[['Class', 'No', 'Name', 'Audio', 'Script', 'Meta', 'Segmented', 'Path']],  rowSelection="single",
                
                enableCellTextSelection=False,
                height=300,
            )
    
    filtered_tutorial = tutorials['selected_rows']
    if filtered_tutorial is not None:
        st.session_state.filtered_tutorial = filtered_tutorial


    filtered_tutorial = st.session_state.get('filtered_tutorial', None)
    # tutorials['selected_rows'] = filtered_tutorial

   
    if filtered_tutorial is not None:
        vid_path = filtered_tutorial['Path'].values[0]
        seg_dir = vid_path.replace('.mp4', '/').replace('tutorials', 'segmented/')
        aud_path = vid_path.replace('.mp4', '.mp3')
        script_path = vid_path.replace('.mp4', '.csv')
        meta_path = vid_path.replace('.mp4', '.json')

        col1, col2, col3 = st.columns([0.30, 0.45, 0.25])

        with col1:
            st.subheader("강의 영상", divider=True)
            st.video(vid_path)

            st.subheader("강의 오디오", divider=True)

            if os.path.exists(aud_path):
                st.audio(aud_path)
            else:
                if st.button("오디오 추출", key='extract_audio'):
                    with st.spinner("Please wait..."):
                        video_clip = VideoFileClip(vid_path)
                        audio_clip = video_clip.audio
                        audio_clip.write_audiofile(aud_path)
                        
                        video_clip.close()
                        audio_clip.close()
                        
                        st.rerun()

        with col2:
            with st.container():
                st.subheader("강의 자막", divider=True)

                if os.path.exists(script_path):
                    df_script = pd.read_csv(script_path)
                    st.dataframe(df_script, hide_index=True, use_container_width=True)
                elif os.path.exists(aud_path):
                    if st.button("자막 추출", key='extract_script'):
                        with st.spinner("Please wait..."):
                            try:
                                segments, info = w_model.transcribe(aud_path, language="ko")
                                
                                with open(script_path, 'w', encoding="utf-8") as f:
                                    f.write('Start,End,Script\n')
                                    for i, segment in enumerate(segments, start=1):
                                        start = f"{segment.start:.2f}"
                                        end = f"{segment.end:.2f}"
                                        text = segment.text.strip()
                                        f.write(f'{start},{end},"{text}"\n', )
                                    f.close()
                            except Exception as e:
                                pass
                        
                        with st.spinner("Please wait..."):
                            df = pd.read_csv(script_path)
                            df['Topic'] = None
                            segments = df['Script'].values.tolist()
                            text_list = [f"{i}. {s}" for i, s in enumerate(segments)]
                            joined = "\n".join(text_list)

                            try:
                            
                                prompt = f"""
                                    다음은 발표에서 연속적으로 말한 문장들입니다. 문장들의 주제를 기준으로 묶어서 구간을 나눠주세요.

                                    출력 형식은 아래와 같이 해주세요:
                                    [
                                    {{ "topic": "인사", "range": [0, 2] }},
                                    {{ "topic": "소개", "range": [3, 5] }},
                                    ...
                                    ]

                                    문장들:
                                    {joined}
                                """

                                result = subprocess.run(
                                    ["ollama", "run", 'mistral'],
                                    input=prompt.encode("utf-8"),
                                    stdout=subprocess.PIPE
                                )

                                result = json.loads(result.stdout.decode("utf-8"))
                                
                                for t in result:
                                    start_idx, end_idx = t['range']
                                    topic = t['topic']
                                    df.loc[start_idx:end_idx, 'Topic'] = topic
                            except Exception as e:
                                pass
                            
                            df.to_csv(script_path, index=False)
                        st.rerun()
                else:
                    st.warning("오디오 파일이 존재하지 않습니다. 오디오를 먼저 추출해주세요.")


        with col3:
            st.subheader("강의 메타데이터", divider=True)
            with st.container(height=500, border=False):
                

                if os.path.exists(meta_path):
                    with open(meta_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        st.json(data)
                elif os.path.exists(script_path):
                    if st.button("메타데이터 생성", key='extract_meta'):
                        with st.spinner("Please wait..."):
                            df = pd.read_csv(script_path)
                            segments = df['Script'].values.tolist()
                            text_list = [f"{i}. {s}" for i, s in enumerate(segments)]
                            joined = "\n".join(text_list)
                            
                            prompt = f"""
                                다음은 한국어 튜토리얼에서 연속적으로 말한 문장들입니다. 아래의 정보를 추출해주세요:

                                1. 강의 제목 (명확히 언급된 경우)
                                2. 강사 이름 (언급된 경우)
                                3. 문장들의 주제에 따라 묶어서 토픽 구간을 나눔 (range는 문장 인덱스 기준)
                                4. 전체 내용을 2~4문장으로 요약

                                출력 형식 예시:
                                {{
                                "lecture_title": "강의 제목",
                                "lecturer": "강사 이름",
                                "topics": [
                                    {{ "topic": "주제1", "range": [0, 3] }},
                                    {{ "topic": "주제2", "range": [4, 7] }}
                                ],
                                "summary": "튜토리얼 전체 요약"
                                }}

                                문장들:
                                {joined}
                            """

                            result = subprocess.run(
                                ["ollama", "run", 'mistral'],
                                input=prompt.encode("utf-8"),
                                stdout=subprocess.PIPE
                            )

                            result = json.loads(result.stdout.decode("utf-8"))
                            
                            with open(meta_path, "w", encoding="utf-8") as f:
                                json.dump(result, f, ensure_ascii=False, indent=2)

                            st.rerun()
                else:
                    st.warning("자막 파일이 존재하지 않습니다. 자막을 먼저 추출해주세요.")

        st.subheader("강의 분할", divider=True)

        if os.path.exists(seg_dir):
            df = load_seg_videos(path=seg_dir).sort_values(by='Start')
            if df is not None:
                col1, col2 = st.columns([0.6, 0.4])
                with col1:
                    segments = AgGrid(df,  rowSelection="single",
                        enableRowSelection=True, 
                        enableCellTextSelection=False,
                        reload_data=True,
                        height=400,
                    )

                    filtered_segment = segments['selected_rows']
                    if filtered_segment is not None:
                        seg_path = filtered_segment['Path'].values[0]
                        col2.video(seg_path)
                    
                    
        elif os.path.exists(script_path):
            df = pd.read_csv(script_path)
            df = df.groupby('Topic').agg({'Start': 'min', 'End': 'max'}).reset_index().sort_values(by='Start')

            df['Start'] = df['Start'].astype(int)
            df['End'] = df['End'].astype(int)
            df['Topic'] = df['Topic'].fillna('None')
            time_ranges = df[['Start', 'End', 'Topic']].values.tolist()
            
           

            if st.button("강의 분할", key='split_video'):
                with st.spinner("Please wait..."):
                    split_video_by_seconds(vid_path, './data/segmented/', time_ranges)
                    
                    st.rerun()
        else:
            st.warning("자막 파일이 존재하지 않습니다. 자막을 먼저 추출해주세요.")