import sys
import os
import sys
import pandas as pd

import streamlit as st
from streamlit import session_state as SS

from molscore.gui.utils import utils
from molscore.gui.utils.pymol_wrapper import PyMol
from molscore.gui.utils.file_picker import st_file_selector
from molscore.gui.utils import single_plot, multi_plot, mpo_plot, scaffold_plot

# ----- Set session state -----
if 'main_df' not in SS.keys(): SS.main_df = None
if 'input_dirs' not in SS.keys(): SS.input_dirs = []
if 'input_latest' not in SS.keys(): SS.input_latest = []
if 'dock_path' not in SS.keys(): SS.dock_path = None
if 'pymol' not in SS: SS.pymol = None
if 'exclude_params' not in SS: SS.exclude_params = ['run', 'dock_path']

# ----- Setup page -----
st.set_page_config(
     page_title='Streamlit monitor',
     layout="wide",
     initial_sidebar_state="expanded",
)


def add_run(input_dir, SS):
    if input_dir not in SS.input_dirs:
        try:
            # Load iterations
            df, latest_idx = utils.load(input_dir=input_dir, latest_idx=0)
             # Add input_dirs
            SS.input_dirs.append(input_dir)
            SS.input_latest.append(latest_idx)
            # Carry index over
            df.reset_index(inplace=True)
            df.rename(columns={'index': 'idx'}, inplace=True)
            # Add main_df
            if SS.main_df is None:
                SS.main_df = df
            else:
                SS.main_df = pd.concat([SS.main_df, df], axis=0, ignore_index=True)
        except FileNotFoundError as e:
            raise e 


def main():
    # ----- Load in data ----
    if len(SS.input_dirs) == 0:
        if (len(sys.argv) == 1):
            temp_selector = st.empty()
            input_dir = st_file_selector(
                st_placeholder=temp_selector,
                path='./',
                label='Please select a molscore output directory',
                key='init_input'                
                )
            temp_msg = st.empty()
            temp_msg.text(input_dir)
            temp_button = st.empty()
            if temp_button.button(label='Load'):
                # Add
                add_run(input_dir=input_dir, SS=SS)
                # Delete file picker and path message
                temp_selector.empty()
                temp_msg.empty()
                temp_button.empty()
        else:
            # Load in any directories in sys.args
            for input_dir in sys.argv[1:]:
                input_dir = os.path.abspath(input_dir)
                add_run(input_dir=input_dir, SS=SS)
        
        # Check any dock path
        SS.dock_path = any([utils.check_dock_paths(d) for d in SS.input_dirs])

    # Setup PyMol
    if (SS.pymol is None) and (SS.dock_path):
        if ('PYMOL_PATH' in os.environ):
            SS.pymol = PyMol(pymol_path=os.environ['PYMOL_PATH'])
        else:
            SS.pymol = False
            print('Export PyMol to PYMOL_PATH to visualise molecules in PyMOl')

    st.sidebar.title('MolScore Monitor')
    if SS.main_df is not None:
        # Header
        if len(SS.input_dirs) > 1:
            st.sidebar.header("Currently loaded:\nMultiple runs")
        else:
            st.sidebar.header(f"Currently loaded:\n{os.path.basename(SS.input_dirs[0])}")

        # Refresh
        st.sidebar.header('Update current runs:')
        if st.sidebar.button('Refresh', help='Check and load any new files molscore runs, useful if monitoring live runs'):
            utils.update(SS=SS)
        
        # Option to add another run
        st.sidebar.header('Add new run:')
        input_dir = st_file_selector(
            st_placeholder=st.sidebar.empty(),
            key='additional_path'
        )
        st.sidebar.write(input_dir)
        if st.sidebar.button('Load', help='Add a seperate molscore run, useful to compare results'):
            add_run(input_dir, SS=SS)
            SS.dock_path = any([utils.check_dock_paths(d) for d in SS.input_dirs])

        # Radio buttons for navigation
        st.sidebar.header('Navigation:')
        nav = st.sidebar.radio(label='Select page', options=['Main', 'Multi-plot', 'MPO', 'Scaffold memory'])

        # ----- Main page -----
        if nav == 'Main':
            single_plot(main_df=SS.main_df, SS=SS, dock_path=SS.dock_path)

        # ----- Multi-plot -----
        if nav == 'Multi-plot':
            multi_plot(main_df=SS.main_df, SS=SS)        

        # ----- MPO Graph -----
        if nav == 'MPO':
            mpo_plot(main_df=SS.main_df, SS=SS, dock_path=SS.dock_path)

        # ----- Scaffold memory page ----
        if nav == 'Scaffold memory':
            scaffold_plot(main_df=SS.main_df, SS=SS, dock_path=SS.dock_path)
        
    # ----- Exit -----
    exit = st.sidebar.button('Exit')
    if exit:
        if SS.pymol is not None:
            SS.pymol.close()
        os._exit(0)


if __name__ == '__main__':
    main()
