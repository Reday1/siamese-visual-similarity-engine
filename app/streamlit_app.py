"""
Premium Streamlit frontend for Siamese Visual Similarity Engine.
Drop-in replacement for app/streamlit_app.py
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
html,body,[data-testid="stAppViewContainer"]{background:#f6f8fc;}
.block-container{padding-top:1.2rem;max-width:1400px;}
.hero{padding:2.2rem;border-radius:22px;background:linear-gradient(135deg,#4f46e5,#06b6d4);
color:#fff;box-shadow:0 18px 40px rgba(0,0,0,.15);margin-bottom:1.4rem;}
.upload{border:2px dashed #b8c3ff;padding:1rem;border-radius:18px;background:#fff;}
.card{background:#fff;border-radius:18px;padding:14px;box-shadow:0 8px 24px rgba(0,0,0,.08);
transition:.25s;}
.card:hover{transform:translateY(-6px);}
img{border-radius:14px;}
</style>
""", unsafe_allow_html=True)

st.markdown("""<div class="hero">
<h1>🐦 Siamese Visual Similarity Engine</h1>
<p>Upload a bird image and retrieve the most visually similar birds using Siamese Neural Network embeddings.</p>
</div>""", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙ Settings")
    k=st.slider("Top Results",1,20,5)
    st.info("Lower distance = higher visual similarity.")
    st.divider()
    st.markdown("**Dataset**\n\nCUB-200-2011")
    st.markdown("**Embedding**\n\nSiamese CNN")

st.markdown('<div class="upload">',unsafe_allow_html=True)
uploaded=st.file_uploader("Upload a JPG or PNG image",type=["jpg","jpeg","png"])
st.markdown("</div>",unsafe_allow_html=True)

if uploaded:
    left,right=st.columns([1,2],gap="large")
    with left:
        st.subheader("Query")
        st.image(uploaded,use_container_width=True)

    suffix=Path(uploaded.name).suffix
    with tempfile.NamedTemporaryFile(delete=False,suffix=suffix) as tmp:
        tmp.write(uploaded.getvalue())
        tmp_path=tmp.name

    try:
        with st.spinner("🧠 Extracting embeddings and searching..."):
            results=retrieve(tmp_path,k=k)
    except Exception as e:
        st.error(f"Could not process the image.\n\nDetails: {e}")
        st.stop()
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    if not results:
        st.warning("No similar images were found.")
        st.stop()

    best=results[0]
    with right:
        a,b,c=st.columns(3)
        a.metric("Results",len(results))
        b.metric("Top Species",best["species"].split(".",1)[-1].replace("_"," "))
        c.metric("Distance",f"{best['distance']:.4f}")

    st.subheader("🏆 Best Match")
    if "url" in best:
        st.image(best["url"],width=500)
    else:
        p=DATA_DIR/best["path"]
        if p.exists():
            st.image(str(p),width=500)
        else:
            st.warning(f"Image not found: {p.name}")
    st.success(f"Distance: {best['distance']:.4f}")

    st.subheader("All Retrieved Matches")
    cols=st.columns(min(5,len(results)))
    for i,item in enumerate(results):
        with cols[i%len(cols)]:
            st.markdown('<div class="card">',unsafe_allow_html=True)
            if "url" in item:
                st.image(item["url"],use_container_width=True)
            else:
                p=DATA_DIR/item["path"]
                if p.exists():
                    st.image(str(p),use_container_width=True)
                else:
                    st.warning(p.name)
            st.markdown(f"**{item['species'].split('.',1)[-1].replace('_',' ')}**")
            st.progress(max(1,min(100,int(max(0,1-item["distance"])*100))))
            st.caption(f"Distance: {item['distance']:.4f}")
            st.markdown("</div>",unsafe_allow_html=True)
else:
    st.info("👆 Upload an image to start retrieval.")

st.divider()
st.caption("Built with Streamlit • Siamese Network • CUB-200-2011")
