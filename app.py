import streamlit as st
import json
from collections import Counter
import matplotlib as mpl
import matplotlib.colors as mcolors
from tue_search import TuebingenSearchEngine as TSE

# -- FUNCTIONS --
@st.cache_data
def load_json(file_path):
    """Load json file with utf-8 encoding."""
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def update_page_title():
    """Update page title according to query."""
    title = "TueSearch"
    if 'input_value' in st.session_state and st.session_state['input_value']:
        title += f": {st.session_state['input_value']}"
    title_placeholder.markdown(f'<h1 class="page-title">{title}</h1>', unsafe_allow_html=True)

def score_to_color(score):
    """Map score to color map."""
    r, g, b = mcolors.to_rgb(cmap(score))
    return f'rgb({int(r * 255)}, {int(g * 255)}, {int(b * 255)})'

@st.cache_resource
def initialize_tse():
    """Pre-load the corpus and retrieval functions."""
    return TSE()

# -- INITIALIZATION --
# set page config
st.set_page_config(page_title="TueSearch", page_icon="🔍", layout="centered")

# load css, json files
with open('static/styles.css') as f:
    css = f.read()
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

# set session_states including required variables and files
session_defaults = {
    'show_results': False,
    'previous_queries': [],
    'current_results': [],
    'input_value': "",
    'pinned_links': [],
    'previous_queries_set': set(),
    'pinned_links_set': set(),
    'tse': initialize_tse(),
    'meta': load_json('data/meta_info.json'),
    'processed_corpus': load_json('data/processed_corpus.json')
}
for key, default in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default

# set variables
cmap = mpl.colormaps['viridis_r']
describtion_limit = 250
keyword_limit = 10
meta = st.session_state['meta']

# -- CONTENT STRUCTURE & HANDLING --
title_placeholder = st.empty()
update_page_title()

st.markdown(
    '<div class="note">Created for MSE by <br>Stephan Amann, Tanja Huber, <br>Markus Potthast, Tina Truong</div>',
    unsafe_allow_html=True
)

# user form
with st.form(key='search_form'):
    query = st.text_input('Enter your search query:')
    # centering the button
    col1, col2, col3= st.columns([4, 2, 4])
    with col2:
        submit_button = st.form_submit_button("Submit")

# handle form submission
if submit_button:
    if query.strip():
        # centering loading spinner
        col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13 = st.columns([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        with col7:
            with st.spinner(""):
                query_result = st.session_state['tse'].search(query)

        if query not in st.session_state['previous_queries_set']:
            st.session_state['previous_queries'].append({
                'query': query,
                'results': query_result
            })
            st.session_state['previous_queries_set'].add(query)

        st.session_state['input_value'] = query
        st.session_state['current_results'] = query_result
        st.session_state['show_results'] = True
        update_page_title()

    else:
        st.write("Please enter a valid search query.")

# main content: search results
if st.session_state['show_results']:
    for result in st.session_state['current_results']:
        doc_id = result[0]
        score = result[1]
        if doc_id in meta:
            col1, col2 = st.columns([9, 2])
            with col1:
                html_content = f"""
                    <div style="margin-bottom: 20px;">
                        <a href='{meta[doc_id]['url']}' class="url" style="color: {score_to_color(score)};">{meta[doc_id]['url']}</a><br>
                        <span class="title">{meta[doc_id]['title']}</span><br>
                    """

                if 'description' in meta[doc_id] and meta[doc_id]['description']:
                    description_text = meta[doc_id]['description']
                    # limit length of description
                    if len(description_text) > describtion_limit:
                        description_text = description_text[:describtion_limit] + '...'
                    html_content += f"<span class='description'>{description_text}</span><br>"

                if 'keywords' in meta[doc_id] and meta[doc_id]['keywords']:
                    html_content += f"<span class='keywords'>{meta[doc_id]['keywords'][:keyword_limit]}</span>"

                else:
                    # gather keywords from processed corpus according to term frequencies
                    cleaned_terms = st.session_state['processed_corpus'][doc_id].split("\n;\n")
                    term_frequencies = Counter(cleaned_terms)
                    filtered_terms = {term: freq for term, freq in term_frequencies.items() if not term.isdigit()}
                    top_terms = [term for term, freq in term_frequencies.most_common(keyword_limit)]
                    keywords_str = ', '.join(top_terms)
                    html_content += f"<span class='keywords'>{keywords_str}</span>"

                html_content += "</div>"
                st.markdown(html_content, unsafe_allow_html=True)

            with col2:
                button_placeholder = st.empty()
                if doc_id not in [d for d, s in st.session_state['pinned_links']]:
                    if button_placeholder.button("Pin", key=f"pin-{doc_id}"):
                        st.session_state['pinned_links'].append((doc_id, score))
                        st.session_state['pinned_links_set'].add(doc_id)
                        button_placeholder.empty()

                else:
                    if button_placeholder.button("Unpin", key=f"unpin-{doc_id}"):
                        st.session_state['pinned_links'] = [
                            (d, s) for d, s in st.session_state['pinned_links'] if d != doc_id
                        ]
                        st.session_state['pinned_links_set'].remove(doc_id)
                        button_placeholder.empty()

        else:
            st.write(f"No meta information found for doc_id: {doc_id}")

# sidebar content: pinned links
with st.sidebar:
    st.sidebar.header("Pinned Links")
    for doc_id, score in st.session_state['pinned_links']:
        if doc_id in meta:
            pinned_link_html = f"""
                <div>
                    <a href='{meta[doc_id]['url']}' class="sidebar-url" style="color: {score_to_color(score)};">{meta[doc_id]['url']}</a><br>
                    <span class="sidebar-title">{meta[doc_id]['title']}</span><br>
                </div>
                """
            st.sidebar.markdown(pinned_link_html, unsafe_allow_html=True)

# sidebar content: previous queries
    st.sidebar.header("Previous Queries")
    if st.session_state['previous_queries']:
        for query_data in st.session_state['previous_queries']:
            if st.button(f"{query_data['query']}"):
                st.session_state['input_value'] = query_data['query']
                st.session_state['current_results'] = query_data['results']
                st.session_state['show_results'] = True
                update_page_title()