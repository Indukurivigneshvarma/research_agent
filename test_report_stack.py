import os
from pprint import pprint

# -----------------------------
# IMPORT REPORT MODULES
# -----------------------------

from report.citations import build_cited_summaries, build_references
from report.headings import generate_title_and_headings
from report.writer import write_report
from report.pdf_generator import generate_pdf


# -----------------------------
# MOCK INPUT DATA (REPLACE IF NEEDED)
# -----------------------------

USER_QUERY = (
    "environmental pollution in New Delhi, India"
)

SUMMARIES = [
    {
        "id": "S1",
        "summary": (
            "Air pollution in India is a significant issue, with 17 of the world's 30 most polluted cities located in the country. New Delhi has the poorest air quality among capital cities globally, with particulate matter (PM2.5) concentrations nearly 10 times higher than World Health Organization guidelines. The main causes of air pollution in India include thermal power plants, vehicle emissions, industrial emissions, burning of crop residue, and burning of wood and dirty fuels for cooking and heating. This results in devastating effects, including over 2 million deaths per year, according to the State of Global Air 2024, and health problems like respiratory and cardiovascular diseases. India's GDP would have increased by $95 billion in 2019 if the country had achieved safe air quality levels, as cleaner air would lead to lower rates of absenteeism from work, higher productivity, and higher consumer footfall. The Indian government has launched the National Clean Air Program to reduce particulate matter pollution by 40% by 2026. Various organizations, including the Indian Institute of Technology Kanpur, the Real Urban Emissions (TRUE) Initiative, and the Confederation of Indian Industry, are working to improve air quality management through data collection, monitoring, and innovation. Businesses also have a significant role to play, with the India CEO Forum for Clean Air and the Corporate Air Emissions Reporting Guide providing opportunities for private sector engagement."
        ),
        "author": "Author One",
        "domain": "example.com",
        "url": "https://example.com/vehicular-emissions",
    },
    {
        "id": "S2",
        "summary": (
            "Air pollution in Delhi has undergone significant changes in terms of pollutant levels and control measures over the past decade. Delhi's air pollution status is characterized by high levels of particulate matter (PM), with the urban air database reporting a maximum PM limit exceeded by almost 10-times at 198 μg/m3 in 2011. Vehicular emissions and industrial activities are major contributors to indoor and outdoor air pollution in Delhi, with vehicular pollution accounting for 67% of air pollutants. A study funded by the World Bank Development Research Group in 1991-1994 found that the average total suspended particulate (TSP) level in Delhi was five-times the World Health Organization's annual average standard, with TSP levels exceeding the World Health Organization's 24-h standard on 97% of days. The study also found that air pollution in Delhi caused more life-years to be lost due to deaths occurring at a younger age. A report by the Ministry of Environment and Forests, India, in 1997 estimated that 3000 metric tons of air pollutants were emitted daily in Delhi, with vehicular pollution contributing 67% and coal-based thermal power plants contributing 12%. The concentrations of carbon monoxide from vehicular emissions increased by 92% from 1989 to 1996. Delhi has the highest cluster of small-scale industries in India, contributing 12% to air pollutants. The PM standard, which includes particles with a diameter of 10 μm or less, is used to measure air quality, with the World Health Organization recommending an annual mean concentration of 20 μg/m3. Exposure to PM has been linked to adverse health effects, including respiratory problems, lung damage, cancer, and premature death, with elderly persons, children, and people with chronic lung disease or asthma being particularly susceptible. The Central Pollution Control Board's 2008 study found significant associations between air pollution and adverse health outcomes, including respiratory symptoms, asthma, and reduced lung function."
        ),
        "author": "Author Two",
        "domain": "example.org",
        "url": "https://example.org/industrial-pollution",
    },
    {
        "id": "S3",
        "summary": (
            "In Delhi and its satellite cities, particulate (PM) pollution is primarily caused by vehicle exhaust, industries, waste burning, and construction activities, with average concentrations of 123±87 μg/m3 for PM2.5 and 208±137 μg/m3 for PM10 between 2008 and 2011, exceeding national annual ambient standards. A multi-sectoral emissions inventory for 2010 estimated health impacts in terms of premature mortality and morbidity effects, with 7,350–16,200 premature deaths and 6.0 million asthma attacks per year. Sector contributions to ambient PM2.5 ranged from 16–34% for vehicle exhaust, 20–27% for diffused sources, 14–21% for industries, 3–16% for diesel generator sets, and 4–17% for brick kilns. The Government of Delhi has implemented initiatives such as the compressed natural gas (CNG) switch for public transport, introduction of new CNG buses, metro system, and Bharat-IV stage fuel, but their marginal benefits have fallen short over the years due to increasing passenger vehicles, lack of public transport, and growing demand for electricity and construction activities."
        ),
        "author": "Author Three",
        "domain": "healthjournal.org",
        "url": "https://healthjournal.org/air-pollution-health",
    },
    {
        "id": "S4",
        "summary": (
            "Delhi's air pollution peaks during winter months, with substantial contributions from vehicle exhaust, road dust, construction dust, cooking and heating, open waste burning, light and heavy industries, diesel generator sets, and seasonal sources. Continuous ambient air quality monitoring expanded from manual to real-time stations, measuring pollutants such as SO2, NOx, CO, NH3, and ozone. The judiciary system has played a critical role in addressing the air pollution problem, with over 100 studies documenting the rise and fall of Delhi's air quality and the interventions that worked and failed to address the issue. Delhi's air quality ranked the worst among the world's capital cities in 2022, with a mix of pollutants exceeding Indian standards and contributing to health impacts. The city's geography and meteorology, including its location on the Indo-Gangetic Plain and surrounding satellite cities, contribute to its air pollution levels, with seasonal meteorology playing a dominant role in exacerbating pollution levels. Air quality is a major social and political concern in India, with limited consensus on source contributions despite numerous studies. Continuous technical and economic interventions, including judicial engagement, have contributed to changes in Delhi's air pollution levels, but Delhi remains the most polluted capital city in the world."
        ),
        "author": "Author Four",
        "domain": "policyreview.net",
        "url": "https://policyreview.net/delhi-air-policy",
    },
]


# -----------------------------
# STEP 1: PREPARE SUMMARY TEXTS
# -----------------------------

summary_texts = [s["summary"] for s in SUMMARIES]

summary_map = {
    s["id"]: s["summary"]
    for s in SUMMARIES
}


# -----------------------------
# STEP 2: CITATIONS
# -----------------------------

print("\n=== TEST: CITATIONS ===")

citations = build_cited_summaries(SUMMARIES)
references = build_references(SUMMARIES)

pprint(citations)
pprint(references)


# -----------------------------
# STEP 3: TITLE + HEADINGS (LLM)
# -----------------------------

print("\n=== TEST: TITLE + HEADINGS ===")

outline = generate_title_and_headings(
    user_query=USER_QUERY,
    summaries=summary_texts,
    max_topics=3,
)

pprint(outline)

title = outline["title"]
headings = outline["headings"]


# -----------------------------
# STEP 4: REPORT WRITING (LLM)
# -----------------------------

print("\n=== TEST: REPORT WRITING ===")

report_text = write_report(
    title=title,
    headings=headings,
    summaries=summary_map,
    references=references,
)

print("\n--- GENERATED REPORT TEXT ---\n")
print(report_text)


# -----------------------------
# STEP 5: PDF GENERATION
# -----------------------------

print("\n=== TEST: PDF GENERATION ===")

pdf_path = generate_pdf(
    report_text=report_text,
    output_path="test_report.pdf",
)

print(f"\n✅ PDF generated at: {pdf_path}")


# -----------------------------
# DONE
# -----------------------------

print("\n🎉 REPORT STACK TEST COMPLETE")
