import streamlit as st
from src.plot import list_all, distribution_histogram, distribution_boxplot, count_Y

def data_visualization(DF):
    st.divider()
    st.subheader('Data Visualization')

    _, col_mid, _ = st.columns([1, 3, 1])
    with col_mid:
        plot_area = st.empty()
        
    col1, col2 = st.columns(2)
    with col1:
        attributes = DF.columns.tolist()
        att = st.selectbox(
            label = 'Select an attribute to visualize:',
            options = attributes,
            index = len(attributes)-1
        )
        st.write(f'Attribute selected: :green[{att}]')
        
    with col2:
        plot_types = ['Distribution histogram', 'Distribution boxplot', 'Donut chart']
        plot_type = st.selectbox(
            label = 'Select a plot type:',
            options = plot_types,
            index = 0
        )
        st.write(f'Plot type selected: :green[{plot_type}]')

    if st.button('Visualize'):
        if plot_type == 'Distribution histogram':
            fig = distribution_histogram(DF, att)
            plot_area.pyplot(fig)
        elif plot_type == 'Distribution Boxplot':
            fig = distribution_boxplot(DF, att)
            plot_area.pyplot(fig)
        elif plot_type == 'Donut chart':
            fig = count_Y(DF, att)
            plot_area.plotly_chart(fig)
    
    st.subheader('Data Overview')
    if 'data_origin' not in st.session_state:
        st.session_state.data_origin = DF
    st.dataframe(st.session_state.data_origin.describe(), width=1200)
    if 'overall_plot' not in st.session_state:
        st.session_state.overall_plot = list_all(st.session_state.data_origin)
    st.pyplot(st.session_state.overall_plot)