# Enhanced version with Best Match section
import os,sys,tempfile
from pathlib import Path
import streamlit as st

PROJECT_ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(PROJECT_ROOT/"src"))
from retrieval import retrieve

DATA_DIR=PROJECT_ROOT/"data"/"CUB_200_2011"
st.set_page_config(page_title="Visual Similarity Engine",page_icon="🐦",layout="wide")

st.markdown("""
<style>
html,body,[data-testid="stAppViewContainer"]{
background:linear-gradient(-45deg,#fdfbfb,#eef5ff,#e8f8ff,#f5ecff);
background-size:400% 400%;animation:bg 18s ease infinite;}
@keyframes bg{0%{background-position:0 50%}50%{background-position:100% 50%}100%{background-position:0 50%}}
.hero{padding:2rem;border-radius:22px;background:linear-gradient(135deg,#5B86E5,#36D1DC);color:#fff;box-shadow:0 10px 30px rgba(0,0,0,.15)}
.glass{background:rgba(255,255,255,.82);backdrop-filter:blur(12px);padding:16px;border-radius:18px;box-shadow:0 8px 24px rgba(0,0,0,.08);margin-top:1rem}
.card{background:white;border-radius:18px;padding:12px;box-shadow:0 8px 24px rgba(0,0,0,.08);transition:.3s}
.card:hover{transform:translateY(-6px)}
img{border-radius:16px}
.stProgress>div>div{background:linear-gradient(90deg,#5B86E5,#36D1DC)!important}
</style>
""",unsafe_allow_html=True)

st.markdown("<div class='hero'><h1>🐦 Siamese Visual Similarity Engine</h1><p>Find visually similar birds using Siamese embeddings.</p></div>",unsafe_allow_html=True)
with st.sidebar:
    k=st.slider("Top Results",1,20,5)
uploaded=st.file_uploader("Upload Bird Image",type=["jpg","jpeg","png"])
if uploaded:
    suf=Path(uploaded.name).suffix
    with tempfile.NamedTemporaryFile(delete=False,suffix=suf) as t:
        t.write(uploaded.getvalue());tmp=t.name
    try:
        with st.spinner("Searching..."):
            results=retrieve(tmp,k=k)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
    best=results[0]
    st.markdown("## 🏆 Best Match")
    l,r=st.columns([1.2,1])
    with l:
        if "url" in best: st.image(best["url"],use_container_width=True)
        else:
            p=DATA_DIR/best["path"]
            if p.exists(): st.image(str(p),use_container_width=True)
    with r:
        species=best["species"].split(".",1)[-1].replace("_"," ")
        sim=max(0,min(100,int((1-best["distance"])*100)))
        st.markdown("<div class='glass'>",unsafe_allow_html=True)
        st.subheader(species)
        st.metric("Similarity",f"{sim}%")
        st.metric("Distance",f"{best['distance']:.4f}")
        st.progress(sim)
        st.success("⭐ Most Similar Bird Found")
        st.markdown("</div>",unsafe_allow_html=True)

    st.markdown("## 📸 All Matches")
    cols=st.columns(min(5,len(results)))
    for i,item in enumerate(results):
        with cols[i%len(cols)]:
            st.markdown("<div class='card'>",unsafe_allow_html=True)
            if "url" in item: st.image(item["url"],use_container_width=True)
            else:
                p=DATA_DIR/item["path"]
                if p.exists(): st.image(str(p),use_container_width=True)
            s=item["species"].split(".",1)[-1].replace("_"," ")
            st.write("**"+s+"**")
            sim=max(0,min(100,int((1-item["distance"])*100)))
            st.progress(sim)
            st.caption(f"Similarity: {sim}%")
            st.caption(f"Distance: {item['distance']:.4f}")
            st.markdown("</div>",unsafe_allow_html=True)
else:
    st.info("Upload an image to begin.")
