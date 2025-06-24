import os
from docling.document_converter import DocumentConverter

source = "C:\\Users\\raninduh\\Downloads\\asset_condition_assesment\\asset_condition_assesment\\2024\\reports_asset_condition_assessment_2024_Appendix C_ACA-ETB 2024.xlsx"  # document per local path or URL
converter = DocumentConverter()
result = converter.convert(source)

# Extract the filename from the source path
filename = os.path.basename(source)
# Remove the file extension and use it for the markdown file
markdown_filename = os.path.splitext(filename)[0] + ".md"

# Get the markdown content
markdown_content = result.document.export_to_markdown()

# Save the markdown content to a file
with open(markdown_filename, 'w', encoding='utf-8') as f:
    f.write(markdown_content)

print(f"Markdown file created: {markdown_filename}")