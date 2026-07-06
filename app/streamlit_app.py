
"""
Premium Animated Streamlit UI (Frontend Only)
Replace your streamlit_app.py with this and merge if needed.
"""

# NOTE:
# This template preserves the uploaded image preview, adds an animated
# light theme, best-match section, CSS animations and modern cards.
# It expects your existing retrieve() function.

import streamlit as st
import tempfile, os, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT/"src"))
from retrieval import retrieve

DATA_DIR = PROJECT_ROOT/"data"/"CUB_200_2011"

st.set_page_config("Visual Similarity Engine","🐦",layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

html,body,[data-testid="stAppViewContainer"]{
font-family:Inter,sans-serif;
background:linear-gradient(-45deg,#ffffff,#eef5ff,#f8fbff,#f4ecff);
background-size:400% 400%;
animation:bgMove 15s ease infinite;
}

@keyframes bgMove{
0%{background-position:0% 50%;}
50%{background-position:100% 50%;}
100%{background-position:0% 50%;}
}

.block-container{
animation:fade .8s ease;
}

@keyframes fade{
from{opacity:0;transform:translateY(20px);}
to{opacity:1;transform:translateY(0);}
}

.hero{
padding:35px;
border-radius:24px;
background:linear-gradient(135deg,#5B86E5,#36D1DC,#8EC5FC);
background-size:200% 200%;
animation:hero 8s ease infinite,float 6s ease-in-out infinite;
color:white;
box-shadow:0 20px 45px rgba(0,0,0,.15);
margin-bottom:25px;
}

@keyframes hero{
0%{background-position:0 50%;}
50%{background-position:100% 50%;}
100%{background-position:0 50%;}
}

@keyframes float{
50%{transform:translateY(-5px);}
}

.card{
background:rgba(255,255,255,.82);
backdrop-filter:blur(18px);
padding:15px;
border-radius:22px;
box-shadow:0 15px 35px rgba(0,0,0,.08);
transition:.35s;
}

.card:hover{
transform:translateY(-8px) scale(1.02);
}

img{
border-radius:18px!important;
transition:.35s;
}

img:hover{
transform:scale(1.04);
}

.stProgress>div>div{
background:linear-gradient(90deg,#5B86E5,#36D1DC)!important;
}
</style>
""",unsafe_allow_html=True)

st.markdown("<div class='hero'><h1>🐦 Siamese Visual Similarity Engine</h1><p>Premium Animated Interface</p></div>",unsafe_allow_html=True)

k=st.sidebar.slider("Top Results",1,20,5)

uploaded=st.file_uploader("📂 Upload Bird Image",type=["jpg","jpeg","png"])

if uploaded:

    st.subheader("📸 Uploaded Image")

    c1,c2=st.columns([1,2])

    with c1:
        st.image(uploaded,use_container_width=True)

    suffix=Path(uploaded.name).suffix

    with tempfile.NamedTemporaryFile(delete=False,suffix=suffix) as tmp:
        tmp.write(uploaded.getvalue())
        temp=tmp.name

    results=retrieve(temp,k=k)

    if os.path.exists(temp):
        os.unlink(temp)

    best=results[0]

    with c2:
        st.subheader("🏆 Best Match")
        species=best["species"].split(".",1)[-1].replace("_"," ")
        st.metric("Species",species)
        st.metric("Distance",f"{best['distance']:.4f}")
        sim=max(0,min(100,int((1-best["distance"])*100)))
        st.progress(sim)
        st.success(f"Similarity : {sim}%")

    st.divider()
    st.subheader("🔎 Retrieved Images")

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
            st.write("**"+item["species"].split(".",1)[-1].replace("_"," ")+"**")
            sim=max(0,min(100,int((1-item["distance"])*100)))
            st.progress(sim)
            st.caption(f"Similarity {sim}%")
            st.caption(f"Distance {item['distance']:.4f}")
            st.markdown("</div>",unsafe_allow_html=True)

else:
    st.info("Upload an image to begin.")
