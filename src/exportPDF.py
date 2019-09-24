import fpdf


def export_pdf(input, spectrum_no, location, name, plot, compounds, prediction, confidence):
    # Create PDF object
    pdf = fpdf.FPDF(format='letter')

    # Add report page
    pdf.add_page()

    # Set PDF attributes
    pdf.set_text_color(4, 22, 51)
    pdf.set_font('Arial', 'B', size=14)

    header(pdf)

    pdf.set_line_width(0.1)
    pdf.set_draw_color(0, 53, 84)
    pdf.line(10, 30, 200, 30)
    pdf.ln(2)
    tab = '\t\t\t\t\t'

    # Get identifier
    spectrum_id = input.split('/')
    spectrum_id = spectrum_id[len(spectrum_id) - 1]

    # Add MS Data plot to PDF
    pdf.cell(200, 10, 'Mass Spectrum (' + spectrum_id + ' #' + str(spectrum_no) + ')', ln=1, align='L')
    plot.write('temp/temp.png', format='png')
    pdf.image('temp/temp.png', w=150, h=80)

    pdf.line(10, 122, 200, 122)
    pdf.ln(2)

    # Add Prediction Results to PDF
    pdf.cell(200, 10, 'Prediction Results', ln=1, align='L')
    pdf.set_font('Arial', size=14)
    pdf.cell(200, 10, tab + 'Pancreatic Cancer: ' + tab + prediction, ln=1, align='L')
    pdf.cell(200, 10, tab + 'Confidence Level: ' + tab + '  ' +confidence, ln=1, align='L')

    pdf.line(10, 160, 200, 160)
    pdf.ln(8)

    # Add chemical compounds list to PDF
    pdf.set_font('Arial', 'B', size=14)
    pdf.cell(100, 10, 'List of Most Abundant Chemical Compounds', ln=0, align='L')
    compounds_list = compounds.split('- ')
    pdf.set_font('Arial', size=14)
    for i in compounds_list:
        pdf.cell(100, 10, tab + i, ln=1, align='L')

    # Output PDF to specified folder location
    pdf.output(location + '\\' + name)


# Page header
def header(self):
    # Logo
    self.image('PDF_banner.png', 10, 8, 190, 18)

    # Move to the right
    self.cell(80)

    # Line break
    self.ln(20)
