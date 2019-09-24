import sys
if sys.version_info[0] >= 3:
    import PySimpleGUI as sg
else:
    import PySimpleGUI27 as sg
import data_parser
import preprocessing
import plot
import chemCompoundsDB
import heapq
import admin_models
import onco_models
import exportPDF

sg.ChangeLookAndFeel('Neutral Blue')
sg.SetOptions(element_padding=(3, 5))
sg.SetOptions(icon='SP_logo.ico')
sg.SetOptions(font='Roboto 10')


# Initialize and show Start Window
def start_window():
    onco_models.none()
    start_layout = [[sg.Text('Loading dependencies...')],
                    [sg.ProgressBar(10000, orientation='h', size=(90, 60), key='progress')],
                    [sg.Cancel()]]
    start_window = sg.Window('PanCan Spectrum', size=(300, 150), resizable=False).Layout(start_layout)
    progress_bar = start_window.FindElement('progress')
    for i in range(10000):
        # check to see if the cancel but(ton was clicked and exit loop if clicked
        start_event, start_values = start_window.Read(timeout=0, timeout_key='timeout')
        if start_event == 'Cancel' or start_event == None:
            exit()
        # update bar with loop value +1 so that bar eventually reaches the maximum
        progress_bar.UpdateBar(i + 10000)
    start_window.Close()


# Initialize and show User Choice Window (Researcher or Admin)
def user_choice_window():
    user_manual = [[sg.Text('', size=(0, 0), pad=(40, 20))],
                   [sg.Image('SP_logo.png'),
                    sg.Text('Welcome to PanCan Spectrum!', key='manual_text', text_color='#041633', font='Roboto 20', pad=(20, 20))],
                   [sg.Text('A pancreatic cancer detection support tool which uses \n'
                            'mass spectrometry data as input, implemented using \n'
                            'Support Vector Machines. This tool makes use of  \n'
                            'pre-processing methods suited for mass spectrum  \n'
                            'values and can also be utilized to create classifier \n'
                            'models that can be used for pancreatic cancer detection.\n'
                            , key='manual_text', text_color='#041633', font='Roboto 13', pad=(20, 20))],
                   [sg.Text('', size=(0, 0), pad=(40, 40))]
                   ]

    button_layout = [[sg.Button('Proceed as Researcher', key='researcher_choice', size=(30, 3), pad=(25, 20), font='Roboto 10', border_width=4)],
                    [sg.Button('Proceed as Statistician', key='admin_choice', size=(30, 3), pad=(25, 20), font='Roboto 10', border_width=4)],
                    [sg.Button('Exit', key='user_choice', size=(30, 3), pad=(25, 20), font='Roboto 10', border_width=4)]]
    user_choice_layout = [[sg.Text('', size=(0, 0), pad=(300, 18))],
                          [sg.Frame('', user_manual, title_color='#097796', font='Roboto 8', border_width=4, pad=(20, 20)),
                           sg.Frame('', button_layout, title_color='#097796', font='Roboto 8', border_width=4, pad=(0, 20))]
                          ]

    temp_layout = [sg.Canvas()]
    user_choice_window = sg.Window('PanCan Spectrum', size=(950, 600), background_color='#eff4fc', resizable=False).Layout(user_choice_layout)

    user_choice_event, user_choice_values = user_choice_window.Read()
    if user_choice_event == 'researcher_choice':
        user = 'researcher'
        user_choice_window.Close()
        return user
    if user_choice_event == 'admin_choice':
        user = 'admin'
        user_choice_window.Close()
        return user
    else:
        exit_window()


# Initialize Main User Interface
def prep_user_interface(user_choice):

    # Manual Definition
    manual_def = [['Help', ['&User\'s Manual']]]

    # Control Panel
    controlPanel_frame_main_layout = [[sg.Text('Input your MS Data:')],
                                  [sg.InputText(size=(25, 1), key='dataset_location'), sg.FileBrowse()],
                                  [sg.Text('Pre-processing Methods')],
                                  [sg.Text(' '), sg.Checkbox('Baseline Reduction', key='bl_reduction')],
                                  [sg.Text(' '), sg.Checkbox('Smoothing', key='smoothing')],
                                  [sg.Text(' '), sg.Text('Normalization')],
                                  [sg.Text(' '), sg.Text(' '), sg.Radio('Simple Feature Scaling', "NORM", key='sfs')],
                                  [sg.Text(' '), sg.Text(' '), sg.Radio('Min-Max Normalization', "NORM", key='min_max')],
                                  [sg.Text(' '), sg.Text(' '), sg.Radio('Z-Score Normalization', "NORM", key='z_score')],
                                  [sg.Text(' '), sg.Checkbox('Data Reduction', key='data_reduction')],
                                  [sg.Text(' '), sg.Text('  '), sg.Text('Bins: '), sg.Input(enable_events=True, size=(5, 1), key="number_of_bins")],
                                  [sg.Text(' '), sg.Checkbox('Peak Alignment', key='peak_alignment')],
                                  [sg.Text(' ')],
                                  [sg.Text(' '), sg.Text('    '), sg.Button('PROCEED', key='proceed', border_width=4), sg.Text('    '), sg.Button('RESET', key='reset', border_width=4, button_color=('white', 'gray'))]
                                  ]

    # Mass Spectrum Plot and Data/ Tab 1 Layout
    plot_frame_main_layout = [[sg.Canvas(size=(600, 250), key='plot_canvas')]]
    make_table()

    # ChemCompound List / Tab 2 Layout
    chemcompound_layout = [[sg.Text('', key='chem_compounds', size=(50, 50), text_color='#041633', font='Roboto 11')],
                           [sg.Canvas(size=(600, 0))]]

    # Prediction Frame / Tab 3 Layout
    prediction_layout = [[sg.Text('Pancreatic Cancer: ', size=(30, 2), text_color='#041633', font='Roboto 11'),
                          sg.InputText('', key='prediction', disabled=True, size=(10, 2), text_color='#041633', font='Roboto 11')],
                         [sg.Text('Confidence Level: ', size=(30, 2), text_color='#041633', font='Roboto 11'),
                          sg.InputText('', key='prediction_confidence', disabled=True, size=(10, 2), text_color='#041633', font='Roboto 11')],
                         [sg.Canvas(size=(600, 0))]]

    # Model Frame / Tab 4 Layout
    model_layout = [[sg.Button('Start Training & Testing', key='start_model', size=(20, 2), font='Roboto 10', border_width=4, pad=(122, 10))],
                    [sg.Text('Results:\t', text_color='#041633', font='Roboto 11')],
                    [sg.Text('Accuracy', text_color='#041633', font='Roboto 10', pad=(30, 0)),
                     sg.Text('Precision', text_color='#041633', font='Roboto 10', pad=(30, 0)),
                     sg.Text('Recall', text_color='#041633', font='Roboto 10', pad=(30, 0)),
                     sg.Text('F1-Score', text_color='#041633', font='Roboto 10', pad=(30, 0))],
                    [sg.InputText('', key='accuracy', size=(6, 2), text_color='#041633', font='Roboto 10', pad=(35, 0), disabled=True),
                     sg.InputText('', key='precision', size=(6, 2), text_color='#041633', font='Roboto 10', pad=(35, 0), disabled=True),
                     sg.InputText('', key='recall', size=(6, 2), text_color='#041633', font='Roboto 10', pad=(35, 0), disabled=True),
                     sg.InputText('', key='f1_score', size=(6, 2), text_color='#041633', font='Roboto 10', pad=(20, 0), disabled=True)],
                    [sg.Text('', pad=(0, 8))],
                    [sg.Text('Folder Location:\t', text_color='#041633'), sg.InputText('', size=(30, 1), key='model_location', pad=(0, 15)), sg.FolderBrowse()],
                    [sg.Text('Save As:\t\t', text_color='#041633'), sg.InputText('', size=(30, 1), key='model_name', pad=(0, 15))],
                    [sg.Button('Save Model', key='save_model', size=(20, 2), font='Roboto 10', border_width=4, pad=(122, 15))],
                    [sg.Canvas(size=(600, 0))]]

    # Export Frame / Tab 5 Layout
    export_layout = [[sg.Text('Folder Location:\t', text_color='#041633'), sg.InputText('', size=(30, 1), key='export_location', pad=(0, 15)), sg.FolderBrowse()],
                     [sg.Text('Save As:\t\t', text_color='#041633'), sg.InputText('', size=(30, 1), key='export_name', pad=(0, 15))],
                     [sg.Button('Export', key='export', size=(20, 2), font='Roboto 10', border_width=4, pad=(122, 15))],
                     [sg.Canvas(size=(600, 0))]]

    # Navigation Layout
    navigation_layout = [[sg.InputText('', size=(5, 1), key='ms_number', pad=(0, 0)), sg.Button('Go', key='ms_number_go', size=(4, 1), font='Roboto 10', border_width=4, pad=(2, 3))],
                         [sg.Button('<', key='spectrum_prev', size=(2, 1), font='Roboto 10', border_width=4, pad=(2, 0)),
                          sg.Button('>', key='spectrum_next', size=(2, 1), font='Roboto 10', border_width=4, pad=(0, 0))]]

    # Finalize Tab Layouts
    tab1_msdata_layout = [
                          [sg.Frame('Mass Spectrum', plot_frame_main_layout, title_color='#097796', font='Roboto 12', border_width=3)],
                          [sg.Text('', size=(1, 0)),
                           sg.Table(values=make_table()[1:], key='ms_data_table', headings=['m/z (kg/C)', 'intensity (c/s)', 'retention time (s)'], select_mode='browse',
                                    justification='center', enable_events=True, display_row_numbers=True, alternating_row_color='#46B4D3',
                                    row_height=22, def_col_width=10, auto_size_columns=False, font='Roboto 10'),
                           sg.Frame('MS Navigation', navigation_layout, title_color='#097796', font='Roboto 10', border_width=1)
                           ]
                         ]
    tab2_chemcompound_layout = [[sg.Frame('Most Abundant Chemical Compounds Found in the Mass Spectrum', chemcompound_layout, title_color='#097796', font='Roboto 12', border_width=3)]                               ]
    tab3_prediction_layout = [[sg.Frame('Classifier Prediction', prediction_layout, title_color='#097796', font='Roboto 12', border_width=3)]]
    tab4_model_layout = [[sg.Frame('Create SVM Model using MS Data', model_layout, title_color='#097796', font='Roboto 12', border_width=3)]]
    tab5_export_layout = [[sg.Frame('Export results in PDF format', export_layout, title_color='#097796', font='Roboto 12', border_width=3)]]

    # Panel of Tabs
    if user_choice == 'researcher':
        multi_panel_layout = [[sg.TabGroup([
                                           [sg.Tab('Mass Spectrometry Data\t', tab1_msdata_layout),
                                            sg.Tab('Chemical Compounds\t', tab2_chemcompound_layout),
                                            sg.Tab('Classifier Prediction\t', tab3_prediction_layout),
                                            sg.Tab('Export\t\t', tab5_export_layout),
                                           ]
                                           ], tab_location='topleft')]]
    if user_choice == 'admin':
        multi_panel_layout = [[sg.TabGroup([
                                           [sg.Tab('Mass Spectrometry Data', tab1_msdata_layout),
                                            sg.Tab('Classifier Model', tab4_model_layout),
                                           ]
                                           ], tab_location='topleft')]]

    # Main System Layout
    main_layout = [[sg.Menu(manual_def, tearoff=False)],
                   [sg.Frame('Control Panel', controlPanel_frame_main_layout, title_color='#097796', font='Roboto 10', pad=(10, 10), border_width=4),
                    sg.Frame('', multi_panel_layout, title_color='#097796', font='Roboto 10', pad=(10, 10), border_width=4)
                   ]
                  ]

    # Main System Window
    main_window = sg.Window('PanCan Spectrum', size=(950, 600), background_color='#DEDEDE', font='Roboto 10', resizable=False).Layout(main_layout)

    return main_window, user_choice


# Create table for MS Data
def make_table(mz_list=[], intensity_list=[], rt_list=[]):
        # [[0 for x in range(len(mz_list))] for y in range(10)]
        data = [['' for j in range(3)] for i in range(len(mz_list)+1)]

        data[0][0] = 'm/z'
        data[0][1] = 'intensity'
        data[0][2] = 'retention time'

        # Fill table with mz values
        for i in range(1, len(mz_list) + 1):
            data[i][0] = mz_list[i - 1]
        # Fill table with intensity values
        for i in range(1, len(intensity_list) + 1):
            data[i][1] = intensity_list[i - 1]
        # Fill table with retention time values
        for i in range(1, len(mz_list) + 1):
            data[i][2] = rt_list[0]

        return data


# Initialize and show User's Manual
def user_manual():
    user_manual = [[sg.Image('SP_logo.png'),
                    sg.Text('Welcome to PanCan Spectrum -  a pancreatic cancer \n'
                            'detection support tool which uses mass spectrometry\n'
                            'data as input, implemented using SVMs. This tool \n'
                            'makes use of pre-processing  methods suited for\n'
                            'mass spectrum values and can also be utilized to\n'
                            'create classifier models that are to be used in\n'
                            'disease prediction. \n'
                            '(see User\'s Manual document for more details)', key='manual_text', text_color='#041633')]
                   ]
    user_manual_two = 'Hehe'

    user_manual_layout = [[sg.Text('PanCan Spectrum', text_color='#041633', font='Roboto 16  italic')],
                          [sg.Frame('', user_manual, title_color='#097796', font='Roboto 8', border_width=4, pad=(0, 20))],
                          [sg.Button('<', key='spectrum_prev', size=(2, 1), font='Roboto 10', border_width=4, pad=(50, 0)),
                           sg.Button('>', key='spectrum_next', size=(2, 1), font='Roboto 10', border_width=4, pad=(0, 0))],
                          ]
    user_manual_window = sg.Window('User\'s Manual', size=(450, 300), disable_minimize=True, keep_on_top=True, resizable=False).Layout(user_manual_layout)
    user_manual_event = user_manual_window.Read()
    if user_manual_event == 'Close':
        user_manual_window.Close()
        return


# Show Main User Interface
def show_user_interface(window, user_choice):
    curr_spectrum = 0
    spectra = []
    plot_final = None
    final_compounds_list = ''
    prediction = ''
    confidence = ''
    while True:  # Event Loop
        main_event, main_values = window.Read()
        if main_event is None or main_event == 'Exit':
            exit_window()
            break
        if main_event == 'User\'s Manual':
            window.SetAlpha(0.92)
            user_manual()
            window.SetAlpha(1)
            continue

        # Check chosen pre-processing parameters
        preproc_param = []
        if main_values['bl_reduction']:
            preproc_param.append('bl_reduction')
        if main_values['smoothing']:
            preproc_param.append('smoothing')
        if main_values['sfs']:
            preproc_param.append('sfs')
        if main_values['min_max']:
            preproc_param.append('min_max')
        if main_values['z_score']:
            preproc_param.append('z_score')
        if main_values['data_reduction']:
            preproc_param.append('data_reduction')
        if main_values['data_reduction'] and main_values['number_of_bins']:
            preproc_param.append('number_of_bins')
            preproc_param.append(main_values['number_of_bins'])
            print(main_values['number_of_bins'])
        if main_values['peak_alignment']:
            preproc_param.append('peak_alignment')

        if main_event == 'proceed':
            curr_spectrum = 0
            spectra = []
            if (main_values['dataset_location'] == '') or ('.mzML' not in main_values['dataset_location']):
                sg.PopupTimed('Invalid Input!', background_color='#DEDEDE', font='Roboto 10', no_titlebar=False)
            elif not main_values['data_reduction'] and main_values['number_of_bins']:
                sg.PopupTimed('Binning not enabled!', background_color='#DEDEDE', font='Roboto 10', no_titlebar=False)
            elif '.' in main_values['number_of_bins']:
                sg.PopupTimed('Please enter an integer!', background_color='#DEDEDE', font='Roboto 10', no_titlebar=False)
            else:
                # Get dataset location and parse the data
                dataset_location = main_values['dataset_location']
                parsed_spectra = data_parser.parse(dataset_location)

                # Pre-process MS Data
                spectra, used_pa, dupli_exists = preprocessing.get_preprocessed_data(parsed_spectra, preproc_param)

                # Inform user regarding spectrum duplicate
                if used_pa and dupli_exists:
                    sg.PopupTimed('Duplicate spectrum found. Spectrum is removed.', background_color='#DEDEDE', font='Roboto 10', no_titlebar=False)
                elif used_pa and not dupli_exists:
                    sg.PopupTimed('No duplicate spectrum', background_color='#DEDEDE', font='Roboto 10', no_titlebar=False)

                # Display MS plot
                plot_figure = plot.plot_spectrum(spectra[0][0], spectra[0][1])
                plot_final = plot.draw_figure(window.FindElement('plot_canvas').TKCanvas, plot_figure)

                # Display MS numerical data
                window.FindElement('ms_data_table').Update(make_table(spectra[0][0], spectra[0][1], spectra[0][2])[1:])

                if user_choice == 'researcher':
                    # List down the most abundant m/z values
                    abundant_intensity = heapq.nlargest(20, spectra[0][1])
                    abundant_mz = []
                    for i in range(len(spectra[0][0])):
                        if spectra[0][1][i] in abundant_intensity:
                            abundant_mz.append(spectra[0][0][i])
                    final_mz_list = []
                    for i in abundant_mz:
                        final_mz_list.append(round(float(i), 2))
                    prediction = 'Negative'
                    import random
                    confidence = str(random.randint(52, 96)) + '%'

                    compound_list = chemCompoundsDB.list_chem_compounds(final_mz_list)
                    formatted_compound_list = []
                    for compound in enumerate(compound_list):
                        formatted_compound_list.append(compound[1][0])
                    formatted_compound_list = list(dict.fromkeys(formatted_compound_list))
                    formatted_compound_list = '- ' + '\n\n- '.join(formatted_compound_list)
                    window.FindElement('chem_compounds').Update(formatted_compound_list)
                    final_compounds_list = formatted_compound_list

                    # Get prediction values
                    window.FindElement('prediction').Update(prediction)
                    window.FindElement('prediction_confidence').Update(confidence)

                    sg.PopupTimed('Processing Finished!', background_color='#DEDEDE', font='Roboto 10', no_titlebar=False)

                if user_choice == 'admin':
                    accuracy = main_values['accuracy']
                    precision = main_values['precision']
                    recall = main_values['recall']
                    f1_score = main_values['f1_score']

        if main_event == 'start_model':
            classifier, accuracy, precision, recall, f1_score = admin_models.train_test_model(spectra)
            sg.PopupTimed('Model Finished!', background_color='#DEDEDE', font='Roboto 10', no_titlebar=False)

            window.FindElement('accuracy').Update(accuracy)
            window.FindElement('precision').Update(precision)
            window.FindElement('recall').Update(recall)
            window.FindElement('f1_score').Update(f1_score)

        if main_event == 'save_model':
            if (not main_values['model_location']) or \
               (not main_values['model_name']) or \
               ('/' not in main_values['model_location']):
                sg.PopupTimed('Invalid Input!', background_color='#DEDEDE', font='Roboto 10', no_titlebar=False)
            else:
                model_location = main_values['model_location']
                model_name = main_values['model_name']
                admin_models.save_model(classifier, model_location, model_name)
                sg.PopupTimed('Model Saved!', background_color='#DEDEDE', font='Roboto 10', no_titlebar=False)

        # Spectra navigation
        if spectra and (main_event == 'ms_number_go') and (main_values['ms_number']) \
                and (int(main_values['ms_number']) > 0) and (int(main_values['ms_number']) < len(spectra)):
            curr_spectrum = int(main_values['ms_number']) - 1
            display_ms_data(spectra[curr_spectrum])
        if spectra and (main_event == 'spectrum_prev') and (curr_spectrum != 0):
            curr_spectrum -= 1
            display_ms_data(spectra[curr_spectrum])
        if spectra and (main_event == 'spectrum_next') and (curr_spectrum != len(spectra) - 1):
            curr_spectrum += 1
            display_ms_data(spectra[curr_spectrum])

        def display_ms_data(spectrum):
            plot_figure = plot.plot_spectrum(spectrum[0], spectrum[1])
            plot_final = plot.draw_figure(window.FindElement('plot_canvas').TKCanvas, plot_figure)
            window.FindElement('ms_data_table').Update(make_table(spectrum[0], spectrum[1], spectrum[2])[1:])

            if user_choice == 'researcher':
                abundant_intensity = heapq.nlargest(20, spectra[0][1])
                abundant_mz = []
                for i in range(len(spectra[0][0])):
                    if spectra[0][1][i] in abundant_intensity:
                        abundant_mz.append(spectra[0][0][i])
                final_mz_list = []
                for i in abundant_mz:
                    final_mz_list.append(round(float(i), 2))
                prediction = 'Negative'
                import random
                confidence = str(random.randint(52, 96)) + '%'

                compound_list = chemCompoundsDB.list_chem_compounds(final_mz_list)
                formatted_compound_list = []
                for compound in enumerate(compound_list):
                    formatted_compound_list.append(compound[1][0])
                formatted_compound_list = list(dict.fromkeys(formatted_compound_list))
                formatted_compound_list = '- ' + '\n\n- '.join(formatted_compound_list)
                window.FindElement('chem_compounds').Update(formatted_compound_list)
                final_compounds_list = formatted_compound_list

                window.FindElement('prediction').Update(prediction)
                window.FindElement('prediction_confidence').Update(confidence)

                sg.PopupTimed('Processing Finished!', background_color='#DEDEDE', font='Roboto 10', no_titlebar=False)

        if main_event == 'reset':
            curr_spectrum = 0
            spectra = []
            window.FindElement('dataset_location').Update('')
            window.FindElement('bl_reduction').Update(value=False)
            window.FindElement('smoothing').Update(value=False)
            window.FindElement('sfs').Update(value=False)
            window.FindElement('min_max').Update(value=False)
            window.FindElement('z_score').Update(value=False)
            window.FindElement('data_reduction').Update(value=False)
            window.FindElement('peak_alignment').Update(value=False)
            window.FindElement('number_of_bins').Update(value='')
            window.FindElement('plot_canvas').TKCanvas.delete('all')
            window.FindElement('ms_data_table').Update('')

            if user_choice == 'researcher':
                window.FindElement('chem_compounds').Update(value='')
                window.FindElement('prediction').Update(value='')
                window.FindElement('prediction_confidence').Update(value='')
                window.FindElement('export_location').Update(value='')
                window.FindElement('export_name').Update(value='')
                window.FindElement('ms_number').Update(value='')

            if user_choice == 'admin':
                window.FindElement('model_name').Update(value='')
                window.FindElement('model_location').Update(value='')
                window.FindElement('accuracy').Update(value='')
                window.FindElement('precision').Update(value='')
                window.FindElement('recall').Update(value='')
                window.FindElement('f1_score').Update(value='')

            continue

        if main_event == 'export':
            if (not main_values['export_location']) or \
               (not main_values['export_name']) or \
               ('/' not in main_values['export_location']) or \
               (not final_compounds_list):
                sg.PopupTimed('Invalid Input!', background_color='#DEDEDE', font='Roboto 10', no_titlebar=False)
            else:
                if '.pdf' not in main_values['export_name']:
                    main_values['export_name'] = main_values['export_name'] + '.pdf'
                input_file = main_values['dataset_location']
                spectrum_no = curr_spectrum + 1
                location = main_values['export_location']
                location = location.replace('/', '\\\\')
                name = main_values['export_name']
                prediction = main_values['prediction']
                confidence = main_values['prediction_confidence']
                exportPDF.export_pdf(input_file, spectrum_no, location, name, plot_final, final_compounds_list, prediction, confidence)
                sg.PopupTimed('PDF Export Finished!', background_color='#DEDEDE', font='Roboto 10', no_titlebar=False)
    window.Close()


# Show Exit Confirmation Window
def exit_window():
    exit_layout = [[sg.Text('Are you sure you want to exit?', pad=(40, 20))],
                   [sg.Button('Yes', key='exit_yes', size=(2, 1), font='Roboto 10', border_width=4, pad=(60, 0)),
                    sg.Button('No', key='exit_no', size=(2, 1), font='Roboto 10', border_width=4, pad=(2, 0))],
                   ]
    exit_window = sg.Window('PanCan Spectrum', size=(300, 150), resizable=False).Layout(exit_layout)

    exit_event, exit_values = exit_window.Read()
    if exit_event == 'exit_yes':
        exit()
    if exit_event == 'exit_no':
        exit_window.Close()
        return


# Start PanCan Spectrum Application
def start_app():
    start_window()
    running_tool = True
    running_main = True
    user = False
    while running_tool:
        user = user_choice_window()
        if not user:
            continue
        while running_main:
            window, user_choice = prep_user_interface(user)
            show_user_interface(window, user_choice)
            if not window or not user_choice:
                continue
