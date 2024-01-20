import PyPDF2
import requests

def download_from_link(url, name = "", extension="pdf"):
    if name == "":
        name = url
    if not name.endswith(extension):
        name += extension
    r = requests.get(url, allow_redirects=True)
    file = open(name, "wb")
    file.write(r.content)
    file.close()

def read_pdf_to_file(input_f, output_f, output_extension=".txt"):
    if not output_f.endswith(output_extension):
        output_f += output_extension
    pdf_file =open(input_f,'rb')
    pdfreader=PyPDF2.PdfReader(pdf_file)
    output_file = open(output_f, "w")
    page_ct = len(pdfreader.pages)
    for i in range(page_ct):
        curr_page = pdfreader.pages[i]
        page_text = curr_page.extract_text()
        output_file.write(page_text)
        output_file.write("\n\n\n\n\n\n")
    output_file.close()
    pdf_file.close()

def read_pdf_to_string(input_f):
    pdf_file =open(input_f,'rb')
    pdfreader=PyPDF2.PdfReader(pdf_file)
    output = ""
    page_ct = len(pdfreader.pages)
    for i in range(page_ct):
        curr_page = pdfreader.pages[i]
        page_text = curr_page.extract_text()
        output += page_text
        output += " "
    return output

def download_and_read_pdf(url, url_fname, output_fname, extension=".pdf",output_extension=".txt"):
    download_from_link(url, url_fname, extension)
    read_pdf_to_file(url_fname, output_fname, output_extension)
def tostring_from_pdf(url, url_fname, extension=".pdf"):
    download_from_link(url, url_fname, extension)
    return read_pdf_to_string(url_fname)

def test():
    download_and_read_pdf("https://arxiv.org/pdf/1706.03762.pdf", "tp.pdf", "tf-text-file")