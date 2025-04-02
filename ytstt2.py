import streamlit as st
import subprocess
import json
import whisper
import os
import torch  # GPU, MPS 확인용

# yt-dlp를 이용하여 유튜브 검색 결과(영상 URL, 제목, 썸네일 등)를 가져오는 함수
def search_videos(query, max_results=5):
    """
    yt-dlp 명령어:
      --dump-json : 결과를 JSON 형태로 출력
      ytsearch{N}: 유튜브에서 검색 결과를 N개 가져옴
    """
    cmd = f'yt-dlp --dump-json "ytsearch{max_results}:{query}"'
    output = subprocess.check_output(cmd, shell=True).decode("utf-8")
    
    results = []
    for line in output.strip().split("\n"):
        data = json.loads(line)
        video_title = data["title"]
        video_url = data["webpage_url"]
        # 썸네일 URL 가져오기 (없으면 None)
        video_thumbnail = data.get("thumbnail", None)
        results.append({
            "title": video_title,
            "url": video_url,
            "thumbnail": video_thumbnail
        })
    return results

# 선택된 유튜브 영상을 오디오(mp3)로 다운로드
def download_audio(url, output_name="temp.mp3"):
    """
    yt-dlp 명령어:
      -x : 오디오만 추출
      --audio-format mp3 : 오디오 포맷을 mp3로 지정
      -o "temp.%(ext)s" : 출력 파일 이름 패턴 지정 (temp)
    """
    cmd = f'yt-dlp -x --audio-format mp3 {url} -o "temp.%(ext)s"'
    subprocess.run(cmd, shell=True)

    # 다운로드 완료 후 "temp.mp3"가 생성되었는지 확인
    if os.path.exists("temp.mp3"):
        return "temp.mp3"
    else:
        return None

# Whisper 모델을 이용해 음성을 텍스트로 변환
def transcribe_audio(model, audio_file):
    result = model.transcribe(audio_file)
    return result["text"]

def detect_device():
    """CUDA, MPS, CPU 중 사용 가능한 디바이스를 자동으로 선택"""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def main():
    st.title("🎙️ YouTube 검색 및 STT 앱")
    st.write("유튜브 검색어를 입력하고, 선택한 영상을 Whisper로 STT하여 txt 파일로 저장합니다.")

    query = st.text_input("🔎 검색어를 입력하세요.", value="")

    if "results" not in st.session_state:
        st.session_state["results"] = []
    
    # 검색 버튼 클릭 시, 유튜브에서 영상 검색
    if st.button("🔍 검색"):
        if not query.strip():
            st.warning("검색어를 입력하세요.")
        else:
            with st.spinner("유튜브 검색 중..."):
                st.session_state["results"] = search_videos(query, max_results=5)
            # 이전에 선택한 영상 초기화
            if "selected_video" in st.session_state:
                del st.session_state["selected_video"]

    # 검색 결과가 있을 때 썸네일과 제목을 함께 출력하고, 선택할 수 있도록 함
    if st.session_state["results"]:
        st.write("### 검색 결과")
        for i, video in enumerate(st.session_state["results"]):
            col1, col2 = st.columns([1, 4])
            with col1:
                if video["thumbnail"]:
                    st.image(video["thumbnail"], width=120)
                else:
                    st.write("썸네일 없음")
            with col2:
                st.write(video["title"])
                if st.button("영상 선택", key=f"select_{i}"):
                    st.session_state["selected_video"] = video

    # 선택한 영상이 있을 경우 표시 및 STT 실행
    if "selected_video" in st.session_state:
        video = st.session_state["selected_video"]
        st.write("**선택한 영상**:", video["title"])
        
        if st.button("▶️ STT 시작"):
            with st.spinner("🎧 오디오 다운로드 중..."):
                audio_path = download_audio(video["url"])
            
            if not audio_path:
                st.error("오디오 다운로드에 실패했습니다.")
                return
            
            with st.spinner("📦 Whisper 모델 로딩 중..."):
                device = detect_device()
                st.info(f"디바이스: `{device}` 에서 모델 로드 중...")
                model = whisper.load_model("base", device=device)
            
            with st.spinner("📝 음성을 텍스트로 변환 중..."):
                text = transcribe_audio(model, audio_path)
            
            st.success("🎉 변환 완료!")
            st.text_area("📄 STT 결과", text, height=300)
            
            if st.button("💾 TXT 파일로 저장"):
                with open("transcription.txt", "w", encoding="utf-8") as f:
                    f.write(text)
                st.success("✅ transcription.txt 파일이 생성되었습니다!")

if __name__ == "__main__":
    main()
