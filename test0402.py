# streamlit run streamlit250519/test0402.py     
import os
import glob
import json
from moviepy import VideoFileClip
import time
import pandas as pd
import streamlit as st
from faster_whisper import WhisperModel
import torch
import subprocess
torch.classes.__path__ = []


def load_videos(path='./data/'):
    vid_files = glob.glob(f'{path}*.mp4')
    
    vids = []
    for vid_file in vid_files:
        vid_name = vid_file.rsplit('\\', 1)[-1]
        
        aud_name = None if not os.path.exists(vid_file.replace('.mp4', '.mp3')) else vid_file.replace('.mp4', '.mp3')
        scr_name = None if not os.path.exists(vid_file.replace('.mp4', '.csv')) else vid_file.replace('.mp4', '.csv')
        meta_name = None if not os.path.exists(vid_file.replace('.mp4', '.json')) else vid_file.replace('.mp4', '.json')
        
        vids.append([vid_name, vid_file, aud_name, scr_name, meta_name])
    

    df = pd.DataFrame(vids, columns=['Name', 'Path', 'Audio', 'Script', 'Meta'])
    return df
    
@st.cache_resource
def load_whisper_model():
    model = WhisperModel("medium", device= "cpu")
    return model


if __name__ == '__main__':
    st.set_page_config(layout="wide")
    st.header("Video Analysis")
    
    w_model = load_whisper_model()
    
    st.subheader('Dataset', divider=True)
    df = load_videos(path='./data/tutorials/')
    
    
    tutorials = st.dataframe(df[['Name', 'Audio', 'Script', 'Meta']], 
                 hide_index=True,
                 on_select="rerun",
                 use_container_width=True, selection_mode='single-row')
    
    video = tutorials.selection.rows
    
    
    
    if video:
        filtered_tutorial = df.iloc[video]
        vid_path = filtered_tutorial['Path'].values[0]
        aud_path = vid_path.replace('.mp4', '.mp3')
        script_path = vid_path.replace('.mp4', '.csv')
        
        col1, col2, col3, col4 = st.columns(4)
        _, col5, _ = st.columns([0.25, 0.5, 0.25])
        
        with col1:
            st.subheader("Video", divider=True)
            if st.button("Play", key='play_video'):
                with col5:
                    st.subheader(f"Playing: {vid_path}", divider=True)
                    st.video(vid_path)
        
        with col2:
            st.subheader("Audio", divider=True)
            if os.path.exists(aud_path):
                if st.button("Play", key='play_audio'):
                    with col5:
                        st.subheader(f"Playing: {aud_path}", divider=True)
                        st.audio(aud_path)
            else:
                if st.button("Extract", key='extract_audio'):
                    with col5:
                        st.subheader(f"Extacting: {aud_path}", divider=True)
                        with st.spinner("Please wait..."):
                            video_clip = VideoFileClip(vid_path)
                            audio_clip = video_clip.audio
                            audio_clip.write_audiofile(aud_path)
                            
                            video_clip.close()
                            audio_clip.close()
                            
                            st.rerun()
        
        with col3:
            st.subheader("Script", divider=True)
            
            if os.path.exists(script_path):
                if st.button("Show", key='show_script'):
                    df_script = pd.read_csv(script_path)
                    
                    with col5:
                        st.dataframe(df_script, hide_index=True, use_container_width=True)
            else:
                if st.button("Extract", key='extract_script'):
                    with col5:
                        st.subheader(f"Extacting: {script_path}", divider=True)
                        with st.spinner("Please wait..."):
                            segments, info = w_model.transcribe(aud_path, language="ko")
                            
                            with open(script_path, 'w', encoding="utf-8") as f:
                                f.write('Start,End,Script\n')
                                for i, segment in enumerate(segments, start=1):
                                    start = f"{segment.start:.2f}"
                                    end = f"{segment.end:.2f}"
                                    text = segment.text.strip()
                                    f.write(f'{start},{end},"{text}"\n', )
                                f.close()
                        
                        with st.spinner("Please wait..."):
                            df = pd.read_csv(script_path)
                            segments = df['Script'].values.tolist()
                            text_list = [f"{i}. {s}" for i, s in enumerate(segments)]
                            joined = "\n".join(text_list)
                            
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
                            df['Topic'] = None
                            for t in result:
                                start_idx, end_idx = t['range']
                                topic = t['topic']
                                df.loc[start_idx:end_idx, 'Topic'] = topic
                            
                            df.to_csv(script_path, index=False)
                        st.rerun()
                       
        with col4:
            st.subheader("Meta", divider=True)
            st.warning("Under development...")
        
        
        