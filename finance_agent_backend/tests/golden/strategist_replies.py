"""Canned Business Strategist judgements for the golden filings.

Every quote is copied from the filing as ingested in the golden DB, so the Strategist's verify rules run on real
evidence. Together the four replies reach each rule: a verified moat, a quote under the wrong chunk id, a
near-verbatim quote, boilerplate and off-topic quotes, a promoted secondary, an unsupported moat downgraded, Wide
kept and Wide capped at Narrow, an explicit "None" moat, known and unknown segments, the threat cap, pricing kept /
replaced / with unsupported figures, concentration kept / code fallback, and matched, fuzzy-matched and unknown
product lines.
"""

from typing import Any, Dict, Tuple


def _ev(chunk_id: str, quote: str) -> Dict[str, str]:
    return {"chunk_id": chunk_id, "quote": quote}


NVDA_2026: Dict[str, Any] = {
    "business_summary": (
        "NVIDIA sells accelerated computing platforms: GPUs, CPUs, networking, systems and software for data centers, "
        "gaming, professional visualization and automotive. Data center AI infrastructure drives most revenue."
    ),
    "revenue_architecture": "Hardware platform sales to OEMs, ODMs, CSPs and distributors, plus software licenses.",
    "segment_details": [
        {"name": "Data Center", "description": "Rack-scale AI compute and networking platforms.",
         "growth_drivers": ["Blackwell architecture", "AI model makers"]},
        {"name": "Gaming", "description": "GeForce GPUs for gamers.", "growth_drivers": []},
        {"name": "Networking", "description": "Not a code-extracted revenue line.", "growth_drivers": []},
    ],
    "primary_moat": {
        "type": "Network Effects",
        "rationale": "A growing developer base and installed base make the platform more valuable to each customer.",
        "evidence": [_ev(
            "dc0cd4c0-7912-4d37-8902-41ac49c6c1e6",
            "The large and growing number of developers and installed base across our platforms strengthens our "
            "ecosystem and increases the value of our platform for our customers.",
        )],
    },
    "secondary_moats": [
        {
            "type": "High Switching Costs",
            "rationale": "Developers are trained on NVIDIA platforms.",
            # Near-verbatim: "in-person and" dropped, cited under its own chunk.
            "evidence": [_ev(
                "7fb4c320-73a6-4bdd-8fcd-c36382daee5e",
                "Our Deep Learning Institute provides online training for developers in industries and organizations "
                "around the world to build AI and accelerated computing applications that leverage our platforms.",
            )],
        },
        {
            "type": "Intangible Assets / Brand",
            "rationale": "Export licenses.",
            # Risk boilerplate ("may not be able") never counts as moat evidence.
            "evidence": [_ev(
                "fca9c280-c26e-4658-9371-97bb61a07be0",
                "In the event that we are able to sell licensed products into the China market, we may not be able to "
                "pass along all or any of the tariff to our customers, and may be subject to litigation, increased "
                "costs, and a harmed competitive position.",
            )],
        },
    ],
    "segment_moats": [
        {
            "segment_name": "Compute & Networking",
            "type": "Cost Advantage",
            "rationale": "Data-center-scale interconnect of compute nodes.",
            "evidence": [_ev(
                "4470942a-c4a2-4d10-95db-a7616d1dad25",
                "This has enabled us to architect data center-scale computing platforms that can interconnect up to "
                "hundreds of thousands of compute nodes with high-performance networking.",
            )],
        },
        {
            "segment_name": "Robotics",
            "type": "Network Effects",
            "rationale": "Not a reportable segment.",
            "evidence": [_ev(
                "91e6ed47-bb4a-4059-b039-64e9b7b41a42",
                "Our AI technology leadership is reinforced by our large and expanding ecosystem.",
            )],
        },
    ],
    "moat_durability": "Wide",
    "moat_trajectory": "Expanding",
    "competitive_threats": [
        {"threat": "Existing competitors and new entrants may offer lower-priced or better-performing products.",
         "evidence": [_ev(
             "c2e8c08d-732f-43d5-89ac-14e514f824e3",
             "We expect competition to increase from both existing competitors and new market entrants with products "
             "that may be lower priced than ours or may provide better performance or additional features not "
             "provided by our products.",
         )]},
        {"threat": "China-based competitors are favoured by the Chinese government.",
         "evidence": [_ev(
             "ab4bd743-97fa-4c4d-94ea-561415c83bdf",
             "The Chinese government has encouraged customers to purchase from our China-based competitors and "
             "discouraged customers from purchasing, importing, or using our data center products, including any "
             "China-specific product designed to comply with U.S. export controls.",
         )]},
        {"threat": "Quantum computing will replace GPUs.",
         "evidence": [_ev("c2e8c08d-732f-43d5-89ac-14e514f824e3",
                          "Quantum computers will make our GPUs obsolete within two years.")]},
    ],
    "pricing_power": {
        "assessment": "Cheaper competing products have lowered selling prices for some NVIDIA products.",
        "evidence": [_ev(
            "c7fb38ce-1a30-418e-92aa-8b7b1ad184a9",
            "Our competitors’ products, services and technologies, including those mentioned above in this Annual "
            "Report on Form 10-K, may be cheaper or provide better functionality or features than ours",
        )],
    },
    "customer_concentration": {
        "disclosure": "Two direct customers represented 22% and 14% of fiscal year 2026 revenue.",
        "evidence": [_ev(
            "f0b359b4-acbd-4784-9d96-e8f55faef355",
            "For fiscal year 2026, sales to one direct customer represented 22% of total revenue and sales to another "
            "direct customer represented 14% of total revenue",
        )],
    },
}

AAPL_2025: Dict[str, Any] = {
    "business_summary": (
        "Apple designs and sells smartphones, personal computers, tablets, wearables and accessories, and a growing "
        "portfolio of services across a large installed base of devices."
    ),
    "revenue_architecture": "Product sales plus services: subscriptions, AppleCare, payments and advertising.",
    "segment_details": [
        {"name": "iPhone", "description": "Smartphone line.", "growth_drivers": ["Pro models"]},
        {"name": "Wearables", "description": "Watch, AirPods and home devices.", "growth_drivers": []},
        {"name": "Vision Pro", "description": "Not a code-extracted revenue line.", "growth_drivers": []},
    ],
    "primary_moat": {
        "type": "Network Effects",
        "rationale": "Brand quote that does not describe a network effect.",
        # Real quote, off-topic for Network Effects: the verified secondary is promoted.
        "evidence": [_ev(
            "553617a0-f3ea-4426-93bf-7468ce6124ce",
            "The Company offers a portfolio of fee-based service and support products under the AppleCare ® brand.",
        )],
    },
    "secondary_moats": [
        {
            "type": "Intangible Assets / Brand",
            "rationale": "Branded service and support products.",
            "evidence": [_ev(
                "553617a0-f3ea-4426-93bf-7468ce6124ce",
                "The Company offers a portfolio of fee-based service and support products under the AppleCare ® "
                "brand.",
            )],
        },
    ],
    "segment_moats": [
        {
            "segment_name": "Services",
            "type": "High Switching Costs",
            "rationale": "Services is a revenue line, not one of the geographic segments.",
            "evidence": [_ev(
                "553617a0-f3ea-4426-93bf-7468ce6124ce",
                "The Company also offers digital content through subscription-based services, including Apple Arcade "
                "® , a game service;",
            )],
        },
    ],
    "moat_durability": "Wide",
    "moat_trajectory": "Stable",
    "competitive_threats": [
        {"threat": "Competitors cut prices and lower margins.",
         "evidence": [_ev(
             "ccb4933e-79d4-40ee-8949-41bd36c1fb12",
             "Competition has been particularly intense as competitors have aggressively cut prices and lowered "
             "product margins.",
         )]},
        {"threat": "Third-party accessories.",
         "evidence": [_ev("6c75bbae-8a74-4c4e-95b5-7b73c195d505",
                          "Accessories includes Apple-branded and third-party accessories.")]},
    ],
    "pricing_power": {
        "assessment": "Apple has pricing power, shown by a richer mix of premium Pro models.",
        "evidence": [_ev(
            "8b8bfe8f-a58b-4bf6-847f-b4d4922daa4e",
            "iPhone net sales increased during 2025 compared to 2024 due to higher net sales of Pro models.",
        )],
    },
    "customer_concentration": {"disclosure": "Not explicitly disclosed in 10-K", "evidence": []},
}

AMZN_2025: Dict[str, Any] = {
    "business_summary": (
        "Amazon runs online and physical stores, a third-party seller marketplace, advertising, Prime subscriptions "
        "and AWS cloud services."
    ),
    "revenue_architecture": "Retail product sales, seller fees, advertising, subscriptions and AWS usage fees.",
    "segment_details": [
        {"name": "AWS", "description": "Cloud infrastructure and AI services.", "growth_drivers": "AI workloads"},
        {"name": "Advertising", "description": "Sponsored, display and video ads.", "growth_drivers": []},
    ],
    "primary_moat": {
        "type": "Cost Advantage",
        "rationale": "Operational excellence.",
        # Real quote that names no cost mechanism, and no secondary survives: the moat is downgraded.
        "evidence": [_ev(
            "7e22c26e-85ca-46a4-97a2-be41983d13dc",
            "We are guided by four principles: customer obsession rather than competitor focus, passion for invention, "
            "commitment to operational excellence, and long-term thinking.",
        )],
    },
    "secondary_moats": [
        {
            "type": "Network Effects",
            "rationale": "Selection attracts customers.",
            "evidence": [_ev(
                "df55e5fa-7077-4431-a6ec-dee1bc3a5291",
                "We believe that the principal competitive factors in our retail businesses include selection, price, "
                "and convenience, including fast and reliable fulfillment.",
            )],
        },
    ],
    "segment_moats": [
        {
            "segment_name": "AWS",
            "type": "Cost Advantage",
            "rationale": "Shared infrastructure serves internal and external demand.",
            # Exact quote cited under the wrong chunk id: it is re-attributed to the chunk that holds it.
            "evidence": [_ev(
                "facb3430-728b-4563-9e60-c5d1e64ef133",
                "Costs to operate our AWS segment are primarily classified as “Technology and infrastructure” as we "
                "leverage a shared infrastructure that supports both our internal technology requirements and "
                "external sales to AWS customers.",
            )],
        },
    ],
    "moat_durability": "Wide",
    "moat_trajectory": "Stable",
    "competitive_threats": [
        {"threat": "Technology lowers barriers for new and smaller competitors.",
         "evidence": [_ev(
             "df55e5fa-7077-4431-a6ec-dee1bc3a5291",
             "The internet and other technologies including artificial intelligence facilitate competitive entry and "
             "comparison shopping, which enhances the ability of new, smaller, or lesser-known businesses to compete "
             "against us.",
         )]},
        {"threat": "Drone delivery start-ups.",
         "evidence": [_ev("df55e5fa-7077-4431-a6ec-dee1bc3a5291",
                          "Drone delivery start-ups threaten our fulfillment network in every market.")]},
    ],
    "pricing_power": {
        "assessment": "Amazon cut international store prices by 37.5% to win share.",
        "evidence": [_ev(
            "30ed127a-3b0c-413e-a178-78443de3cf2a",
            "This guidance includes approximately $1 billion of higher year-over-year Amazon Leo costs as we scale in "
            "2026, as well as investment in quick commerce and even sharper prices in our international stores "
            "business.",
        )],
    },
    "customer_concentration": {
        "disclosure": "No customer accounted for more than 10% of net sales.",
        "evidence": [_ev("df55e5fa-7077-4431-a6ec-dee1bc3a5291",
                         "No single customer accounted for 10% or more of our net sales in any year.")],
    },
}

TSLA_2025: Dict[str, Any] = {
    "segment_details": [
        {"name": "Automotive sales", "description": "", "growth_drivers": "Model Y and Cybertruck"},
        {"name": "Energy storage", "description": "Not a code-extracted revenue line.", "growth_drivers": []},
        "not a dict",
    ],
    "primary_moat": {"type": "None", "rationale": "The filing evidences no structural competitive advantage.",
                     "evidence": []},
    "secondary_moats": [
        {
            "type": "Efficient Scale",
            "rationale": "Few solar competitors.",
            "evidence": [_ev("7c85c7fc-8363-43e8-ad8b-d0e1b24db4f8",
                             "We also compete with solar energy companies that provide products and services similar "
                             "to ours.")],
        },
    ],
    "segment_moats": [
        {
            "segment_name": "Energy generation and storage",
            "type": "Cost Advantage",
            "rationale": "In-house inverter improves efficiency.",
            "evidence": [_ev(
                "a2a74734-2e9e-49fe-b5f7-4b322c22c552",
                "The efficiency of our energy generation products is aided by our own solar inverter, which "
                "incorporates our power electronics technologies.",
            )],
        },
    ],
    "moat_durability": "Narrow",
    "moat_trajectory": "Expanding",
    "competitive_threats": [
        {"threat": "The automotive market is becoming more competitive.",
         "evidence": [_ev("86efef80-0b53-4339-b583-4ecd28500cf2",
                          "The worldwide automotive market is highly competitive today and we expect it will become "
                          "even more so in the future.")]},
        # Near-verbatim: "also" dropped.
        {"threat": "Competition in AI and robotics.",
         "evidence": [_ev("2d21d18f-c4e4-4f7e-8d5e-83666d47e9dd",
                          "As we seek to become a top provider of autonomous solutions, we face competition in the "
                          "fields of AI and robotics.")]},
        {"threat": "Established and new manufacturers are entering the EV market.",
         "evidence": [_ev("2d21d18f-c4e4-4f7e-8d5e-83666d47e9dd",
                          "We believe that our vehicles compete in the market based on both their traditional segment "
                          "classification as well as their propulsion technology.")]},
        {"threat": "Energy storage rivals offer similar or alternative products.",
         "evidence": [_ev("7c85c7fc-8363-43e8-ad8b-d0e1b24db4f8",
                          "We compete with these companies based on price, energy density, efficiency, optimization "
                          "and performance.")]},
        {"threat": "Competition could cut unit sales and prices.",
         "evidence": [_ev("86efef80-0b53-4339-b583-4ecd28500cf2",
                          "Increased competition could result in our lower vehicle unit sales, price reductions, "
                          "revenue shortfalls, loss of customers and loss of market share, which may harm our "
                          "business, financial condition and operating results.")]},
    ],
    "pricing_power": {"assessment": "Not explicitly disclosed in 10-K", "evidence": []},
    "customer_concentration": {"disclosure": "Tesla has no meaningful customer concentration.", "evidence": []},
}

STRATEGIST_REPLIES: Dict[Tuple[str, int], Dict[str, Any]] = {
    ("AAPL", 2025): AAPL_2025,
    ("AMZN", 2025): AMZN_2025,
    ("NVDA", 2026): NVDA_2026,
    ("TSLA", 2025): TSLA_2025,
}
