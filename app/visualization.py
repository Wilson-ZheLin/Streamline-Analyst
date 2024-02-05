import streamlit as st
from src.plot import list_all, distribution_histogram, distribution_boxplot, count_Y, box_plot, violin_plot, strip_plot, density_plot ,multi_plot_heatmap, multi_plot_scatter, multi_plot_line

def data_visualization(DF):
    st.divider()
    st.subheader('Data Visualization')
    attributes = DF.columns.tolist()

    single_tab, multiple_tab = st.tabs(['Single Attribute Visualization', 'Multiple Attributes Visualization'])
    with single_tab:
        _, col_mid, _ = st.columns([1, 5, 1])
        with col_mid:
            plot_area = st.empty()
            
        col1, col2 = st.columns(2)
        with col1:
            att = st.selectbox(
                label = 'Select an attribute to visualize:',
                options = attributes,
                index = len(attributes)-1
            )
            st.write(f'Attribute selected: :green[{att}]')
            
        with col2:
            plot_types = ['Donut chart', 'Violin plot', 'Distribution histogram', 'Boxplot', 'Density plot', 'Strip plot', 'Distribution boxplot']
            plot_type = st.selectbox(
                key = 'plot_type1',
                label = 'Select a plot type:',
                options = plot_types,
                index = 0
            )
            st.write(f'Plot type selected: :green[{plot_type}]')

        if plot_type == 'Distribution histogram':
            fig = distribution_histogram(DF, att)
            plot_area.pyplot(fig)
        elif plot_type == 'Distribution boxplot':
            fig = distribution_boxplot(DF, att)
            if fig == -1:
                plot_area.error('The attribute is not numeric')
            else:
                plot_area.pyplot(fig)
        elif plot_type == 'Donut chart':
            fig = count_Y(DF, att)
            plot_area.plotly_chart(fig)
        elif plot_type == 'Boxplot':
            fig = box_plot(DF, [att])
            plot_area.plotly_chart(fig)
        elif plot_type == 'Violin plot':
            fig = violin_plot(DF, [att])
            plot_area.plotly_chart(fig)
        elif plot_type == 'Strip plot':
            fig = strip_plot(DF, [att])
            plot_area.plotly_chart(fig)
        elif plot_type == 'Density plot':
            fig = density_plot(DF, att)
            plot_area.plotly_chart(fig)

    with multiple_tab:
        col1, col2 = st.columns([6, 4])
        with col1:
            options = st.multiselect(
                label = 'Select multiple attributes to visualize:',
                options = attributes,
                default = []
            )
        with col2:
            plot_types = ["Violin plot", "Boxplot", "Heatmap", "Strip plot", "Line plot", "Scatter plot"]
            plot_type = st.selectbox(
                key = 'plot_type2',
                label = 'Select a plot type:',
                options = plot_types,
                index = 0
            )
        _, col_mid, _ = st.columns([1, 5, 1])
        with col_mid:
            plot_area = st.empty()

        if options:
            if plot_type == 'Scatter plot':
                fig = multi_plot_scatter(DF, options)
                if fig == -1:
                    plot_area.error('Scatter plot requires two attributes')
                else:
                    plot_area.pyplot(fig)
            elif plot_type == 'Heatmap':
                fig = multi_plot_heatmap(DF, options)
                if fig == -1:
                    plot_area.error('The attributes are not numeric')
                else:
                    plot_area.pyplot(fig)
            elif plot_type == 'Boxplot':
                fig = box_plot(DF, options)
                if fig == -1:
                    plot_area.error('The attributes are not numeric')
                else:
                    plot_area.plotly_chart(fig)
            elif plot_type == 'Violin plot':
                fig = violin_plot(DF, options)
                if fig == -1:
                    plot_area.error('The attributes are not numeric')
                else:
                    plot_area.plotly_chart(fig)
            elif plot_type == 'Strip plot':
                fig = strip_plot(DF, options)
                if fig == -1:
                    plot_area.error('The attributes are not numeric')
                else:
                    plot_area.plotly_chart(fig)
            elif plot_type == 'Line plot':
                fig = multi_plot_line(DF, options)
                if fig == -1:
                    plot_area.error('The attributes are not numeric')
                elif fig == -2:
                    plot_area.error('Line plot requires two attributes')
                else:
                    plot_area.pyplot(fig)
    
    st.subheader('Data Overview')
    if 'data_origin' not in st.session_state:
        st.session_state.data_origin = DF
    st.dataframe(st.session_state.data_origin.describe(), width=1200)
    if 'overall_plot' not in st.session_state:
        st.session_state.overall_plot = list_all(st.session_state.data_origin)
    st.pyplot(st.session_state.overall_plot)

    st.divider()
    st.write(":grey[Streamline Analyst is developed by *Zhe Lin*. You can reach out to me via] :blue[wilson.linzhe@gmail.com] :grey[or] :blue[[GitHub](https://github.com/Wilson-ZheLin)]")