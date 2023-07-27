import streamlit as st
import numpy as np
import torch
import torch.nn as nn

from sentence_transformers import SentenceTransformer
from loguru import logger

def logits_to_labels(y, category_mappings, top_N: int=3):
    top_N_idx = np.apply_along_axis(lambda i: (-i).argsort()[:top_N], axis=1, arr=y)
    top_N_vals = np.apply_along_axis(lambda i: i[(-i).argsort()[:top_N]], axis=1, arr=y)
    
    labels = []

    for doc_top_indeces, doc_top_confidences in zip(top_N_idx, top_N_vals):
        doc_top_N_names = [category_mappings[x] for x in doc_top_indeces ] 
        doc_top_N_vals =  [float(i) for i in doc_top_confidences ] 

        labels.append([[i, j] for i, j in zip(doc_top_N_names, doc_top_N_vals)])

    return labels

@st.cache_resource
def load_llm_model(name):
    model = SentenceTransformer(name) 
    model.eval()
    return model
    
llm_mini = load_llm_model("all-MiniLM-L6-v2")
llm_mpnet = load_llm_model("paraphrase-mpnet-base-v2")

@st.cache_resource
def load_clf(name):
    return torch.jit.load(name)

clf_mini = load_clf('classifier_all-MiniLM-L6-v2.pt')
clf_mpnet = load_clf('classifier_paraphrase-mpnet-base-v2.pt')



def proc():
    encode = st.session_state['llm'].encode([st.session_state.text_key])

    X_t = torch.tensor(encode, dtype=torch.float32)

    y_raw = st.session_state['clf'](X_t)
    # rectify the negative entries, which point strongly against belonging to the corresponding class
    y = nn.ReLU()(y_raw).detach().numpy()

    category_mapping = {
        0: "arts, culture and entertainment",
        1: "automotive",
        2: "crime, law and justice",
        3: "disaster and accident",
        4: "economy, business and finance",
        5: "education",
        6: "environment",
        7: "health",
        8: "labour",
        9: "lifestyle",
        10: "news",
        11: "non-standard content",
        12: "pets",
        13: "politics",
        14: "religion and belief",
        15: "science and technology",
        16: "society",
        17: "sport",
        18: "unrest, conflicts and war"
    }

    y_labels = logits_to_labels(y, category_mapping)

    st.session_state.results_key = str(y_labels)
    
    

def main():
    st.title("Hello, there! Let's classify some text, shall we?")

    st.header("Paste your plain text below and hit GO")

    txt = st.text_area('hide me', """The World Cup co-host looked to have earned itself a route back into the game midway through the second half when Jacqui Hand’s looping header floated over a despairing Olivia McDaniel in goal, but it was later ruled out by the video assistant referee (VAR) for offside. The World Cup debutant was able to withstand New Zealand pressure, including a truly remarkable diving save from McDaniel in added time at the end of the game, to earn a historic victory, sparking scenes of jubilant celebrations.
        """, label_visibility='hidden', key='text_key', placeholder='Enter plain text')
    
    btn = st.button('Go')
    
    if btn == True:
        proc()

    st.text_area('Classification', '',key='results_key')
    # btnGo = st.button('GO', on_click=classify, args=(txt, ))


    llm_options = ["all-MiniLM-L6-v2", "paraphrase-mpnet-base-v2"]
    selected_llm = st.sidebar.selectbox("Choose a language model", llm_options)

    if selected_llm == "all-MiniLM-L6-v2":
        st.session_state['llm'] = llm_mini
        st.session_state['clf'] = clf_mini
        
    elif selected_llm == "paraphrase-mpnet-base-v2":
        st.session_state['llm'] = llm_mpnet
        st.session_state['clf'] = clf_mpnet
        

if __name__ == "__main__":
    main()