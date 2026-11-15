"""Creates fictional sample data: 3 PDFs + units.csv (stand-in for a Google Sheet).
Run:  python data/make_sample_data.py"""
import csv
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, PageBreak

HERE = Path(__file__).parent
(HERE / "pdfs").mkdir(exist_ok=True)
ss = getSampleStyleSheet()


def make_pdf(name, title, pages):
    story = [Paragraph(title, ss["Title"]), Spacer(1, 12)]
    for i, page in enumerate(pages):
        for heading, text in page:
            story += [Paragraph(heading, ss["Heading2"]), Paragraph(text, ss["BodyText"]), Spacer(1, 8)]
        if i < len(pages) - 1:
            story.append(PageBreak())
    SimpleDocTemplate(str(HERE / "pdfs" / name), pagesize=A4).build(story)


make_pdf("Palm_Residence_Brochure.pdf", "Palm Residence - New Cairo (Fifth Settlement)", [
    [("Overview", "Palm Residence is a gated residential compound in New Cairo, Fifth Settlement, "
      "built by Palm Developments. Expected delivery is December 2028. The compound is 40% green space."),
     ("Unit types", "Apartments from 1 to 3 bedrooms (65 to 165 sqm) and townhouses of 4 bedrooms (240 sqm). "
      "Finishing is semi-finished with core and shell options available.")],
    [("Payment plan", "Standard plan: 10% down payment, then equal quarterly installments over 8 years. "
      "A 5% discount applies for full cash payment. The maintenance deposit is 8% of the unit price, "
      "paid at delivery."),
     ("Amenities", "Clubhouse, two swimming pools, gym, kids area, jogging track, mosque, "
      "retail strip and 24/7 security with gated access.")],
])

make_pdf("Nile_View_Towers_Brochure.pdf", "Nile View Towers - Sheikh Zayed", [
    [("Overview", "Nile View Towers is a mid-rise residential project in Sheikh Zayed City by Nile Star Real Estate. "
      "Expected delivery is June 2027. The project offers fully finished units only."),
     ("Unit types", "Studios (55 sqm), 1-bedroom (80 sqm), 2-bedroom (115 sqm) and 3-bedroom (150 sqm) apartments.")],
    [("Payment plan", "Standard plan: 15% down payment and monthly installments over 6 years. "
      "A delivery payment of 5% of the unit price is due at handover. "
      "The maintenance deposit is 7% of the unit price."),
     ("Amenities", "Rooftop terrace, residents' gym, underground parking, community garden and 24/7 security.")],
])

make_pdf("Sales_Policy.pdf", "Sales and Reservation Policy", [
    [("Reservation", "A reservation requires a fee of 50,000 EGP. The fee is fully refundable if the client "
      "cancels within 14 days of paying it."),
     ("Cancellation", "After 14 days, cancellation by the client incurs a 10% deduction from the amounts paid.")],
    [("Resale and transfer", "Units can be resold only with written developer approval. "
      "A transfer fee of 2% of the unit price is charged to the seller."),
     ("Site visits", "Viewings are by appointment only, Sunday to Thursday, 10 AM to 6 PM. "
      "Bookings are made through the assigned sales agent.")],
])

rows = [
    # Unit ID, Project, Location, Type, Bedrooms, Area (sqm), Price (EGP), Status, Delivery Year
    ("PR-101", "Palm Residence", "New Cairo", "Apartment", 1, 65, 3200000, "Available", 2028),
    ("PR-102", "Palm Residence", "New Cairo", "Apartment", 2, 110, 5100000, "Available", 2028),
    ("PR-103", "Palm Residence", "New Cairo", "Apartment", 2, 125, 5750000, "Reserved", 2028),
    ("PR-104", "Palm Residence", "New Cairo", "Apartment", 3, 165, 7900000, "Available", 2028),
    ("PR-105", "Palm Residence", "New Cairo", "Apartment", 3, 150, 7200000, "Sold", 2028),
    ("PR-201", "Palm Residence", "New Cairo", "Townhouse", 4, 240, 14500000, "Available", 2028),
    ("PR-202", "Palm Residence", "New Cairo", "Townhouse", 4, 240, 14900000, "Available", 2028),
    ("NV-101", "Nile View Towers", "Sheikh Zayed", "Studio", 0, 55, 2600000, "Available", 2027),
    ("NV-102", "Nile View Towers", "Sheikh Zayed", "Apartment", 1, 80, 3600000, "Available", 2027),
    ("NV-103", "Nile View Towers", "Sheikh Zayed", "Apartment", 1, 80, 3750000, "Sold", 2027),
    ("NV-201", "Nile View Towers", "Sheikh Zayed", "Apartment", 2, 115, 5400000, "Available", 2027),
    ("NV-202", "Nile View Towers", "Sheikh Zayed", "Apartment", 2, 115, 5550000, "Reserved", 2027),
    ("NV-301", "Nile View Towers", "Sheikh Zayed", "Apartment", 3, 150, 7000000, "Available", 2027),
]
with open(HERE / "units.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["Unit ID", "Project", "Location", "Type", "Bedrooms", "Area (sqm)", "Price (EGP)", "Status", "Delivery Year"])
    w.writerows(rows)
print("Sample data written to", HERE)
