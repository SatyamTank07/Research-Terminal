"""Canned Risk Analyst judgements for the golden filings.

Every quote is copied from the filing's Item 1A as ingested in the golden DB, so the Risk Analyst's verify rules
run on real evidence. Together the replies reach each rule: a verified risk, a quote under the wrong chunk id, a
near-verbatim quote, an invented quote (risk dropped), a malformed entry, defaulted and aliased ratings, a coerced
category, impact capped for immateriality / for no stated magnitude / for boilerplate, a High impact kept by a
realized effect, an exposure removed for having no unit and one for unsupported figures, mitigation kept and
removed, a title duplicate and a shared-chunk duplicate merged, the cap of 8 risks, fewer than 5 risks, a primary
threat kept / re-pointed to a more severe risk / re-pointed after it failed verification, New / Expanded /
Unchanged against the prior year (AAPL, AMZN), and the coverage repair pass adding and merging risks (NVDA).

Quotes cite the chunk ids of the golden DB as ingested on 2026-11-01. If the golden filings are re-ingested, the ids
change: re-pick the chunk ids (the quotes stay) or the claimed-chunk cases turn into rejections, which the golden
diff shows.
"""

from typing import Any, Dict, Tuple


def _ev(chunk_id: str, quote: str) -> Dict[str, str]:
    return {"chunk_id": chunk_id, "quote": quote}


def _risk(risk_id: str, category: Any, title: str, summary: str, likelihood: Any, impact: Any, evidence: list,
          transmission: Any = ("Revenue",), exposure: Any = None, mitigation: Any = None,
          mitigation_evidence: Any = None, signposts: Any = ("Disclosure updates in the next 10-K",)) -> Dict[str, Any]:
    return {
        "risk_id": risk_id,
        "risk_category": category,
        "risk_title": title,
        "risk_summary": summary,
        "likelihood": likelihood,
        "impact": impact,
        "financial_transmission": list(transmission) if isinstance(transmission, tuple) else transmission,
        "quantified_exposure": exposure,
        "mitigating_factors": mitigation,
        "mitigation_evidence": mitigation_evidence,
        "evidence": evidence,
        "monitoring_signposts": list(signposts) if isinstance(signposts, tuple) else signposts,
    }


NVDA_2026: Dict[str, Any] = {
    "identified_risks": [
        _risk(
            "R1", "Supply Chain & Concentration", "Direct customer concentration",
            "Two direct customers account for over a third of revenue.", "High", "High",
            [_ev("35105665-2438-49b9-bc56-5b9dfcbe0340",  # wrong chunk id: resolved to the chunk that holds it
                 "For fiscal year 2026, sales to one direct customer represented 22% of total revenue and sales to "
                 "another direct customer represented 14% of total revenue, all of which were primarily attributable "
                 "to the Compute & Networking segment.")],
            exposure="22% and 14% of total revenue",
            signposts=["Customer mix disclosures", "Hyperscaler capex guidance"],
        ),
        _risk(
            "R2", "Regulatory & Legal", "China export controls",
            "USG licensing requirements for H20 led to an inventory and purchase-obligation charge.", "High", "High",
            [_ev("dc5b89ea-d453-4a06-906f-a83eb3262f20",
                 "As a result of these requirements, we incurred a $4.5 billion charge in the first quarter of fiscal "
                 "year 2026 associated with H20 for excess inventory and purchase obligations, as the demand for H20 "
                 "products diminished.")],
            transmission=["Revenue", "Gross Margin"],
            exposure="$4.5 billion charge",
            mitigation="The company seeks to comply strictly with export control regulations.",
            mitigation_evidence=_ev("5258535b-b24f-4bed-bf3e-ead4a79cfd7b",
                                    "While we seek to strictly comply with all applicable export control regulators"),
        ),
        _risk(
            "R3", "Macroeconomic & Geopolitical", "Export controls on China sales",
            "Licensed sales into China face a tariff and uncertain licensing.", "Medium", "High",
            [_ev("dc5b89ea-d453-4a06-906f-a83eb3262f20",
                 "As a result, any H200 shipped under the new licensing program will be subject to a 25% tariff upon "
                 "importation into the United States.")],
            transmission=["Revenue", "terminal value", "FX"],
            signposts=["New BIS rules"],
        ),
        _risk(
            "R4", "Supply Chain & Concentration", "Foundry dependence",
            "Wafers are made by third-party foundries.", "Medium", "High",
            [_ev("e316a55d-81fe-4469-89cc-96b366d21ddb",
                 "We depend on foundries to manufacture our semiconductor wafers using their fabrication equipment and "
                 "techniques.")],
            transmission=["Gross Margin", "Capex & Working Capital"],
            mitigation="Multiple foundry partners",
        ),
        _risk(
            "R5", "Competitive & Demand", "Competitive pricing pressure",
            "Competitor products may be cheaper or better.", "Medium", "Medium",
            # Near-verbatim: "and technologies," dropped.
            [_ev("c7fb38ce-1a30-418e-92aa-8b7b1ad184a9",
                 "Our competitors’ products, services, including those mentioned above in this Annual Report on Form "
                 "10-K, may be cheaper or provide better functionality or features than ours, which has resulted and "
                 "may in the future result in lower-than-expected selling prices or demand for our products.")],
            transmission=["Revenue", "Gross Margin"],
        ),
        _risk(
            "R6", "Financial & Capital Structure", "Stock price volatility",
            "Operating results may fluctuate below analyst expectations and the stock price could decline.",
            "High", "Medium",
            [_ev("3c03af28-4236-48d5-bfe3-6a5ee668fd9b",
                 "Our operating results have in the past fluctuated and may in the future fluctuate, and if our "
                 "operating results are below the expectations of securities analysts or investors, our stock price "
                 "could decline.")],
            transmission=["Cost of Capital"],
        ),
        _risk(
            "R7", "Operational", "Unknown emerging risks",
            "Risks not presently known may harm the business.", "Low", "Low",
            [_ev("84fcacfd-c063-4c4d-b30c-a6064cb99d70",
                 "Additional risks, trends and uncertainties not presently known to us or that we currently believe "
                 "are immaterial may also harm our business, financial condition, results of operations or reputation.")],
        ),
        _risk(
            "R8", "Cybersecurity", "Security breaches and cyber-attacks",
            "Breaches could disrupt operations.", "Medium", "High",
            [_ev("84fcacfd-c063-4c4d-b30c-a6064cb99d70",
                 "Product, system security and data protection incidents or breaches, as well as cyber-attacks could "
                 "disrupt our operations and adversely affect our financial condition, stock price and reputation.")],
            transmission=["Operating Expenses"],
            exposure="$2 billion of breach costs",
        ),
        _risk(
            "R10", "Macroeconomic & Geopolitical", "International revenue exposure",
            "A large share of revenue comes from outside the United States.", "Medium", "Medium",
            [_ev("35105665-2438-49b9-bc56-5b9dfcbe0340",
                 "we generated 31% of our revenue in fiscal year 2026 from sales outside the United States.")],
            exposure="31% of revenue",
        ),
        _risk(
            "R11", "Operational", "Single power grid dependence",
            "All data centers depend on one regional power grid.", "High", "High",
            [_ev("e316a55d-81fe-4469-89cc-96b366d21ddb",
                 "All of our data centers depend on a single power grid in Texas.")],
        ),
        {"risk_id": "R12", "risk_title": "", "risk_summary": "Missing title."},
    ],
    "primary_threat_risk_id": "R5",
    "primary_existential_threat": "Competitors undercut pricing.",
}

NVDA_2026_REPAIR: Dict[str, Any] = {
    "identified_risks": [
        _risk(
            "C1", "Supply Chain & Concentration", "Indirect customer concentration",
            "Some indirect customers individually represent 10% or more of revenue.", "High", "High",
            [_ev("3045bbae-aa3a-4c9d-9ec8-127f188e5207",
                 "We generate a significant amount of our revenue from a limited number of indirect customers, and we "
                 "estimate some individually representing 10% or more of our revenue.")],
            exposure="10% or more of our revenue",
        ),
        _risk(
            "C2", "Regulatory & Legal", "USG revenue share on licensed China sales",
            "The USG expects a share of revenue from licensed China sales.", "Medium", "Medium",
            [_ev("dc5b89ea-d453-4a06-906f-a83eb3262f20",
                 "USG officials expressed an expectation that the USG will receive 15% or more of the revenue "
                 "generated from licensed sales of our products, but the USG did not publish a regulation codifying "
                 "such requirement.")],
            exposure="15% or more of the revenue",
        ),
        "not a risk",
    ],
}

AAPL_2025: Dict[str, Any] = {
    "identified_risks": [
        _risk(
            "R1", "Macroeconomic & Geopolitical", "Tariffs and trade restrictions",
            "New U.S. tariffs and reciprocal tariffs affect the supply chain and costs.", "High", "Medium",
            [
                _ev("220d737f-5be3-4507-8aa0-1a0f7ba8cc77",
                    "Restrictions on international trade, such as tariffs and other controls on imports or exports of "
                    "goods, technology or data, can materially adversely affect the Company’s business and supply "
                    "chain."),
            ],
            transmission=["Gross Margin", "Revenue"],
        ),
        _risk(
            "R2", "Macroeconomic & Geopolitical", "2025 U.S. import duties",
            "New U.S. import duties announced in 2025 raise product costs.", "High", "Medium",
            [_ev("220d737f-5be3-4507-8aa0-1a0f7ba8cc77",
                 "Beginning in the second quarter of 2025, new tariffs were announced on imports to the U.S.")],
            transmission=["Gross Margin"],
        ),
        _risk(
            "R3", "Operational", "Public health crises",
            "Pandemics have hurt the economy and demand for consumer products.", "Medium", "High",
            [_ev("1cd93f71-8577-4dcd-8392-370567136057",
                 "Major public health issues, including pandemics such as the COVID-19 pandemic, have adversely "
                 "affected, and could in the future materially adversely affect, the Company due to their impact on "
                 "the global economy and demand for consumer products")],
        ),
        _risk(
            "R4", "Competitive & Demand", "Google licensing revenue",
            "Search licensing revenue from Google is subject to investigations and proceedings.", "Medium", "Medium",
            [_ev("d45e7df2-7879-492e-afe0-de24d6e72156",
                 "the Company earns revenue from licensing arrangements with Google LLC (“Google”) and other companies "
                 "to offer their search services on the Company’s platforms and applications")],
        ),
        _risk(
            "R5", "Regulatory & Legal", "Google antitrust ruling on licensing payments",
            "Google was found to have violated antitrust laws; remedies may change the arrangements.", "Medium",
            "High",
            [_ev("d45e7df2-7879-492e-afe0-de24d6e72156",
                 "On August 5, 2024, Google was found to have violated U.S.")],
        ),
    ],
    "primary_threat_risk_id": "R3",
    "primary_existential_threat": "",
}

AMZN_2025: Dict[str, Any] = {
    "identified_risks": [
        _risk(
            "R1", "Competitive & Demand", "Intense competition",
            "Competitors may secure better vendor terms and price more aggressively.", "Elevated", "Medium",
            [_ev("494ab046-523e-422a-91dc-1a1a6a94fc1a",
                 "They may secure better terms from vendors, adopt more aggressive pricing, and devote more resources "
                 "to technology, infrastructure, fulfillment, and marketing.")],
            transmission=["Revenue", "Gross Margin"],
        ),
        _risk(
            "R2", "Macroeconomic & Geopolitical", "International operations",
            "International activities are significant to revenues and profits.", "Medium", "Medium",
            [_ev("fe2746d1-346f-4f7c-8359-02bfc49878bb",
                 "Our international activities are significant to our revenues and profits, and we plan to further "
                 "expand internationally.")],
            transmission=["revenue", "FX"],
        ),
        _risk(
            "R3", "Technological & Cybersecurity", "Security incidents",
            "Systems have experienced security incidents.", "Medium", "High",
            [_ev("b432996b-de6c-4141-80db-1067c7750cd1",
                 "Some of our systems have experienced past security incidents, and, although they did not have a "
                 "material adverse effect on our operating results, there can be no assurance that future incidents "
                 "will not have material adverse effects on our operations or financial results.")],
            transmission=["Operating Expenses"],
            mitigation="Insurance coverage",
            mitigation_evidence=_ev("4021acbe-39c3-4c19-add1-4490987a824b",
                                    "In addition, our insurance may not provide sufficient coverage to compensate for "
                                    "related losses."),
        ),
        _risk(
            "R4", "Operational", "Peak season staffing",
            "Fulfillment network staffing may fall short in peak periods.", "Medium", "Low",
            [_ev("7c21fb8a-a616-41b5-96c8-4b424b5d0c81",
                 "we may be unable to adequately staff our fulfillment network and customer service centers during "
                 "these peak periods")],
            transmission=["Operating Expenses"],
        ),
        _risk(
            "R5", "Regulatory & Legal", "Government regulation",
            "Regulation of offerings and competition may change.", "Low", "High",
            [_ev("abb11ac0-3761-4c49-9a20-6a55e917c16f",
                 "government regulation (such as regulation of our product and service offerings and of competition);")],
        ),
        _risk(
            "R6", "Financial & Capital Structure", "Investment impairments",
            "Valuations of investments may decline.", "Medium", "High",
            [_ev("015dd3b2-006d-4676-bf0b-dde7eaa1905f",
                 "We could determine that such valuations have experienced impairments or other-than-temporary "
                 "declines in fair value which could adversely impact our financial results.")],
            transmission=["Balance Sheet & Liquidity"],
            exposure="N/A",
        ),
        _risk(
            "R7", "Macroeconomic & Geopolitical", "Exchange rate fluctuations and market volatility",
            "Exposure to exchange rate fluctuations grows with international operations.", "Medium", "Medium",
            [_ev("d0686cce-007c-433b-884d-d59b88164e63",
                 "As we have expanded our international operations, our exposure to exchange rate fluctuations has "
                 "increased.")],
            transmission=["Revenue", "Operating Expenses"],
        ),
        _risk(
            "R8", "Taxation", "Additional tax liabilities",
            "Changes to tax laws and the global tax framework may add tax expense.", "probable", "Low",
            [_ev("6f2ccaca-556b-4f16-80a9-eb95de43e730",
                 "We may recognize additional tax expense and be subject to additional tax liabilities, including other "
                 "liabilities for tax collection obligations due to changes in laws, regulations, administrative "
                 "practices, principles, and interpretations related to tax")],
            transmission="Operating Expenses",
            exposure="higher tax expense",
            signposts="OECD Pillar Two adoption",
        ),
    ],
    "primary_threat_risk_id": "R99",
    "primary_existential_threat": "",
}

TSLA_2025: Dict[str, Any] = {
    "identified_risks": [
        _risk(
            "R1", "Operational", "Dependence on Elon Musk",
            "The company is highly dependent on its CEO.", "Medium", "High",
            [_ev("baf66bd4-7d64-4805-9025-53517d5286a8",
                 "In particular, we are highly dependent on the services of Elon Musk, Technoking of Tesla and our "
                 "Chief Executive Officer.")],
            transmission=["Terminal Value"],
        ),
        _risk(
            "R2", "Supply Chain & Concentration", "Battery cell supply",
            "Growth requires substantially more lithium-ion cells.", "High", "High",
            [
                _ev("c909ad16-10c0-487f-959d-de25aba49915",
                    "We are dependent on the continued supply of lithium-ion battery cells for our vehicles and energy "
                    "storage products, and we will require substantially more cells to grow our business according "
                    "to our plans."),
                _ev("7e9ac2a0-5437-44fa-93a6-814b9e084782",
                    "We will need to maintain and significantly grow our access to battery cells, including through "
                    "the development and manufacture of our own cells, and control our related costs."),
            ],
            transmission=["Gross Margin", "Capex & Working Capital"],
        ),
        _risk(
            "R3", "Technological & Cybersecurity", "Autonomy and Robotaxi execution",
            "The future business depends on autonomous driving and Cybercab.", "High", "High",
            [_ev("0f01ed4c-dbf2-4ace-9abb-9726562025c4",
                 "our future business depends on development of our driver assistance systems and autonomous driving "
                 "solutions and increasing the production of mass-market vehicles, including Cybercab, our "
                 "purpose-built Robotaxi product.")],
            transmission=["Revenue", "Terminal Value"],
        ),
        _risk(
            "R4", "Regulatory & Legal", "Claims and scrutiny over driver assistance",
            "The company faces claims and regulatory scrutiny over its technologies.", "High", "High",
            [_ev("45a19053-3a2e-41e8-b6cd-1f1622eaa6be",
                 "We have experienced, and we expect to continue to face, claims and regulatory scrutiny arising from "
                 "or related to misuse or claimed failures or alleged misrepresentations of the technologies that we "
                 "are pioneering.")],
            transmission=["Operating Expenses", "Revenue"],
        ),
        _risk(
            "R5", "Competitive & Demand", "Loss of EV incentives",
            "Expired EV incentives may have hurt sales.", "Medium", "High",
            [_ev("0d95bcbc-2e74-4824-8e19-3fcdf8aa2e64",
                 "previously available incentives favoring electric vehicles in certain areas have expired or were "
                 "cancelled or temporarily unavailable, and in some cases were not eventually replaced or "
                 "reinstituted, which may have negatively impacted sales.")],
        ),
        _risk(
            "R6", "Competitive & Demand", "Vehicle market competition",
            "Target demographics are highly competitive.", "Medium", "Medium",
            [_ev("21a45a2e-c0e9-471e-962c-d9ab05e7ef23",
                 "The target demographics for our vehicles are highly competitive.")],
        ),
    ],
    "primary_threat_risk_id": "R4",
    "primary_existential_threat": "Claims and regulatory scrutiny could limit FSD and Robotaxi.",
}

RISK_REPLIES: Dict[Tuple[str, int], Dict[str, Any]] = {
    ("AAPL", 2025): AAPL_2025,
    ("AMZN", 2025): AMZN_2025,
    ("NVDA", 2026): NVDA_2026,
    ("TSLA", 2025): TSLA_2025,
}

RISK_REPAIR_REPLIES: Dict[Tuple[str, int], Dict[str, Any]] = {
    ("NVDA", 2026): NVDA_2026_REPAIR,
}
