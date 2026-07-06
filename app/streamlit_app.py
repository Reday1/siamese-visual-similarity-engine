"""
Premium Streamlit frontend for Siamese Visual Similarity Engine.
Animated Light UI
"""
import os, sys, tempfile
from pathlib import Path
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from retrieval import retrieve

DATA_DIR = PROJECT_ROOT / "data" / "CUB_200_2011"

st.set_page_config(page_title="Visual Similarity Engine", page_icon="🐦", layout="wide")

st.markdown("""
<style>
html,body,[data-testid="stAppViewContainer"]{
background:linear-gradient(-45deg,#fdfbfb,#eef4ff,#e8f7ff,#f5ecff);
background-size:400% 400%;
animation:gradientBG 18s ease infinite;
}
@keyframes gradientBG{
0%{background-position:0% 50%;}
50%{background-position:100% 50%;}
100%{background-position:0% 50%;}
}
.block-container{max-width:1400px;padding-top:1rem;animation:fadeUp .8s ease;}
@keyframes fadeUp{from{opacity:0;transform:translateY(18px);}to{opacity:1;transform:translateY(0);}}

.hero{
padding:2.2rem;
border-radius:24px;
background:linear-gradient(135deg,#667eea,#764ba2,#89f7fe);
background-size:200% 200%;
animation:heroMove 8s ease infinite,float 6s ease-in-out infinite;
color:white;
box-shadow:0 18px 45px rgba(0,0,0,.15);
margin-bottom:1.5rem;
}
@keyframes heroMove{
0%{background-position:0% 50%;}
50%{background-position:100% 50%;}
100%{background-position:0% 50%;}
}
@keyframes float{
50%{transform:translateY(-6px);}
}

.upload,.card,[data-testid="metric-container"]{
background:rgba(255,255,255,.72);
backdrop-filter:blur(16px);
-webkit-backdrop-filter:blur(16px);
border:1px solid rgba(255,255,255,.45);
border-radius:20px;
box-shadow:0 10px 30px rgba(0,0,0,.08);
transition:.35s;
}

.upload{
border:2px dashed #8aa4ff;
padding:1rem;
}
.upload:hover{
transform:translateY(-4px);
box-shadow:0 20px 45px rgba(91,134,229,.18);
}

.card{padding:14px;}
.card:hover,[data-testid="metric-container"]:hover{
transform:translateY(-8px) scale(1.02);
box-shadow:0 18px 40px rgba(91,134,229,.18);
}

img{
border-radius:16px;
transition:.35s;
}
img:hover{transform:scale(1.03);}

.stProgress>div>div{
background:linear-gradient(90deg,#667eea,#36d1dc)!important;
}

section[data-testid="stSidebar"]{
background:rgba(255,255,255,.75);
backdrop-filter:blur(18px);
}

</style>
""", unsafe_allow_html=True)

st.markdown("""<div class='hero'>
<h1>🐦 Siamese Visual Similarity Engine</h1>
<p>Upload a bird image and retrieve the most visually similar birds using Siamese Neural Network embeddings.</p>
</div>""", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙ Settings")
    k=st.slider("Top Results",1,20,5)
    st.info("Lower distance = higher visual similarity.")
    st.divider()
    st.markdown("**Dataset:** CUB-200-2011")
    st.markdown("**Model:** Siamese CNN")

st.markdown("<div class='upload'>",unsafe_allow_html=True)
uploaded=st.file_uploader("📁 Upload JPG / PNG",type=["jpg","jpeg","png"])
st.markdown("</div>",unsafe_allow_html=True)

if uploaded:
    left,right=st.columns([1,2],gap="large")
    with left:
        st.subheader("Query Image")
        st.image(uploaded,use_container_width=True)

    suffix=Path(uploaded.name).suffix
    with tempfile.NamedTemporaryFile(delete=False,suffix=suffix) as tmp:
        tmp.write(uploaded.getvalue())
        tmp_path=tmp.name

    try:
        with st.spinner("🧠 Extracting embeddings and searching..."):
            results=retrieve(tmp_path,k=k)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    best=results[0]
    with right:
        a,b,c=st.columns(3)
        a.metric("Results",len(results))
        b.metric("Best Match",best["species"].split(".",1)[-1].replace("_"," "))
        c.metric("Distance",f"{best['distance']:.4f}")

    st.subheader("Retrieved Matches")
    cols=st.columns(min(5,len(results)))
    for i,item in enumerate(results):
        with cols[i%len(cols)]:
            st.markdown("<div class='card'>",unsafe_allow_html=True)
            if "url" in item:
                st.image(item["url"],use_container_width=True)
            else:
                p=DATA_DIR/item["path"]
                if p.exists():
                    st.image(str(p),use_container_width=True)
            st.markdown(f"### {item['species'].split('.',1)[-1].replace('_',' ')}")
            sim=max(0,min(100,int((1-item["distance"])*100)))
            st.progress(sim)
            st.caption(f"Similarity: {sim}%")
            st.caption(f"Distance: {item['distance']:.4f}")
            st.markdown("</div>",unsafe_allow_html=True)
else:
    st.info("👆 Upload an image to begin.")

st.divider()
st.caption("Built with ❤️ using Streamlit • PyTorch • Siamese Networks")
