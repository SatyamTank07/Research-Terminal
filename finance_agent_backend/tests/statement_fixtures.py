"""Statement table fixtures in the exact markdown format stored by the ingestion pipeline.

TSLA FY2025 tables reproduce the legacy `Col_n` layout (split `$` cells, colspan drift).
The AAPL-style tables use a different label vocabulary ("Total net sales", "Gross margin",
"Shares used in computing earnings per share", share counts in thousands) to guard
against overfitting the extractor to a single filer.
"""

TSLA_FY2025_INCOME_STATEMENT = """### Year Ended December 31,
| Col_1 | 2025 | Col_3 | 2024 | Col_5 | 2023 | Col_7 | Col_8 | Col_9 | Col_10 | Col_11 | Col_12 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Revenues |  |  |  |  |  |  |  |  |  |  |  |
| Automotive sales | $ | 65,821 |  |  | $ | 72,480 |  |  | $ | 78,509 |  |
| Automotive regulatory credits | 1,993 |  |  | 2,763 |  |  | 1,790 |  |  |  |  |
| Automotive leasing | 1,712 |  |  | 1,827 |  |  | 2,120 |  |  |  |  |
| Total automotive revenues | 69,526 |  |  | 77,070 |  |  | 82,419 |  |  |  |  |
| Energy generation and storage | 12,771 |  |  | 10,086 |  |  | 6,035 |  |  |  |  |
| Services and other | 12,530 |  |  | 10,534 |  |  | 8,319 |  |  |  |  |
| Total revenues | 94,827 |  |  | 97,690 |  |  | 96,773 |  |  |  |  |
| Cost of revenues |  |  |  |  |  |  |  |  |  |  |  |
| Automotive sales | 56,267 |  |  | 61,870 |  |  | 65,121 |  |  |  |  |
| Automotive leasing | 898 |  |  | 1,003 |  |  | 1,268 |  |  |  |  |
| Total automotive cost of revenues | 57,165 |  |  | 62,873 |  |  | 66,389 |  |  |  |  |
| Energy generation and storage | 8,969 |  |  | 7,446 |  |  | 4,894 |  |  |  |  |
| Services and other | 11,599 |  |  | 9,921 |  |  | 7,830 |  |  |  |  |
| Total cost of revenues | 77,733 |  |  | 80,240 |  |  | 79,113 |  |  |  |  |
| Gross profit | 17,094 |  |  | 17,450 |  |  | 17,660 |  |  |  |  |
| Operating expenses |  |  |  |  |  |  |  |  |  |  |  |
| Research and development | 6,411 |  |  | 4,540 |  |  | 3,969 |  |  |  |  |
| Selling, general and administrative | 5,834 |  |  | 5,150 |  |  | 4,800 |  |  |  |  |
| Restructuring and other | 494 |  |  | 684 |  |  | — |  |  |  |  |
| Total operating expenses | 12,739 |  |  | 10,374 |  |  | 8,769 |  |  |  |  |
| Income from operations | 4,355 |  |  | 7,076 |  |  | 8,891 |  |  |  |  |
| Interest income | 1,680 |  |  | 1,569 |  |  | 1,066 |  |  |  |  |
| Interest expense | ( 338 ) |  |  | ( 350 ) |  |  | ( 156 ) |  |  |  |  |
| Other (expense) income, net | ( 419 ) |  |  | 695 |  |  | 172 |  |  |  |  |
| Income before income taxes | 5,278 |  |  | 8,990 |  |  | 9,973 |  |  |  |  |
| Provision for (benefit from) income taxes | 1,423 |  |  | 1,837 |  |  | ( 5,001 ) |  |  |  |  |
| Net income | 3,855 |  |  | 7,153 |  |  | 14,974 |  |  |  |  |
| Net income (loss) attributable to noncontrolling interests and redeemable noncontrolling interests in subsidiaries | 61 |  |  | 62 |  |  | ( 23 ) |  |  |  |  |
| Net income attributable to common stockholders | $ | 3,794 |  |  | $ | 7,091 |  |  | $ | 14,997 |  |
| Net income per share of common stock attributable to common stockholders |  |  |  |  |  |  |  |  |  |  |  |
| Basic | $ | 1.18 |  |  | $ | 2.23 |  |  | $ | 4.73 |  |
| Diluted | $ | 1.08 |  |  | $ | 2.04 |  |  | $ | 4.30 |  |
| Weighted average shares used in computing net income per share of common stock |  |  |  |  |  |  |  |  |  |  |  |
| Basic | 3,225 |  | 3,197 |  | 3,174 |  |  |  |  |  |  |
| Diluted | 3,528 |  | 3,498 |  | 3,485 |  |  |  |  |  |  |"""

TSLA_FY2025_BALANCE_SHEET = """| Col_1 | December 31, 2025 | Col_3 | December 31, 2024 | Col_5 | Col_6 | Col_7 | Col_8 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Assets |  |  |  |  |  |  |  |
| Current assets |  |  |  |  |  |  |  |
| Cash and cash equivalents | $ | 16,513 |  |  | $ | 16,139 |  |
| Short-term investments | 27,546 |  |  | 20,424 |  |  |  |
| Accounts receivable, net | 4,576 |  |  | 4,418 |  |  |  |
| Inventory | 12,392 |  |  | 12,017 |  |  |  |
| Prepaid expenses and other current assets | 7,615 |  |  | 5,362 |  |  |  |
| Total current assets | 68,642 |  |  | 58,360 |  |  |  |
| Operating lease vehicles, net | 4,912 |  |  | 5,581 |  |  |  |
| Energy generation and storage systems, net | 4,604 |  |  | 4,924 |  |  |  |
| Property, plant and equipment, net | 40,643 |  |  | 35,836 |  |  |  |
| Operating lease right-of-use assets | 6,027 |  |  | 5,160 |  |  |  |
| Digital assets | 1,008 |  |  | 1,076 |  |  |  |
| Deferred tax assets | 6,925 |  |  | 6,524 |  |  |  |
| Other non-current assets | 5,045 |  |  | 4,609 |  |  |  |
| Total assets | $ | 137,806 |  |  | $ | 122,070 |  |
| Liabilities |  |  |  |  |  |  |  |
| Current liabilities |  |  |  |  |  |  |  |
| Accounts payable | $ | 13,371 |  |  | $ | 12,474 |  |
| Accrued liabilities and other | 13,279 |  |  | 10,723 |  |  |  |
| Deferred revenue | 3,424 |  |  | 3,168 |  |  |  |
| Current portion of debt and finance leases | 1,640 |  |  | 2,456 |  |  |  |
| Total current liabilities | 31,714 |  |  | 28,821 |  |  |  |
| Debt and finance leases, net of current portion | 6,736 |  |  | 5,757 |  |  |  |
| Deferred revenue, net of current portion | 3,631 |  |  | 3,317 |  |  |  |
| Other long-term liabilities | 12,860 |  |  | 10,495 |  |  |  |
| Total liabilities | 54,941 |  |  | 48,390 |  |  |  |
| Commitments and contingencies (Note 13) |  |  |  |  |  |  |  |
| Redeemable noncontrolling interests in subsidiaries | 58 |  |  | 63 |  |  |  |
| Equity |  |  |  |  |  |  |  |
| Stockholders’ equity |  |  |  |  |  |  |  |
| Preferred stock; $0.001 par value; 100 shares authorized; no shares issued and outstanding | — |  |  | — |  |  |  |
| Common stock; $0.001 par value; 6,000 shares authorized; 3,751 and 3,216 shares issued and outstanding as of December 31, 2025 and 2024, respectively | 3 |  |  | 3 |  |  |  |
| Additional paid-in capital | 42,770 |  |  | 38,371 |  |  |  |
| Accumulated other comprehensive income (loss) | 361 |  |  | ( 670 ) |  |  |  |
| Retained earnings | 39,003 |  |  | 35,209 |  |  |  |
| Total stockholders’ equity | 82,137 |  |  | 72,913 |  |  |  |
| Noncontrolling interests in subsidiaries | 670 |  |  | 704 |  |  |  |
| Total liabilities and equity | $ | 137,806 |  |  | $ | 122,070 |  |"""

TSLA_FY2025_CASH_FLOW = """### Year Ended December 31,
| Col_1 | 2025 | Col_3 | 2024 | Col_5 | 2023 | Col_7 | Col_8 | Col_9 | Col_10 | Col_11 | Col_12 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Cash Flows from Operating Activities |  |  |  |  |  |  |  |  |  |  |  |
| Net income | $ | 3,855 |  |  | $ | 7,153 |  |  | $ | 14,974 |  |
| Adjustments to reconcile net income to net cash provided by operating activities: |  |  |  |  |  |  |  |  |  |  |  |
| Depreciation, amortization and impairment | 6,148 |  |  | 5,368 |  |  | 4,667 |  |  |  |  |
| Stock-based compensation | 2,825 |  |  | 1,999 |  |  | 1,812 |  |  |  |  |
| Inventory and purchase commitments write-downs | 362 |  |  | 335 |  |  | 463 |  |  |  |  |
| Foreign currency transaction net unrealized loss (gain) | 452 |  |  | ( 73 ) |  |  | ( 144 ) |  |  |  |  |
| Deferred income taxes | 123 |  |  | 477 |  |  | ( 6,349 ) |  |  |  |  |
| Non-cash interest and other operating activities | 272 |  |  | 172 |  |  | 81 |  |  |  |  |
| Digital assets loss (gain), net | 68 |  |  | ( 589 ) |  |  | — |  |  |  |  |
| Changes in operating assets and liabilities: |  |  |  |  |  |  |  |  |  |  |  |
| Accounts receivable | ( 261 ) |  |  | ( 1,083 ) |  |  | ( 586 ) |  |  |  |  |
| Inventory | ( 630 ) |  |  | 937 |  |  | ( 1,195 ) |  |  |  |  |
| Operating lease vehicles | ( 25 ) |  |  | ( 590 ) |  |  | ( 1,952 ) |  |  |  |  |
| Prepaid expenses and other assets | ( 3,181 ) |  |  | ( 3,273 ) |  |  | ( 2,652 ) |  |  |  |  |
| Accounts payable, accrued and other liabilities | 4,376 |  |  | 3,588 |  |  | 2,605 |  |  |  |  |
| Deferred revenue | 363 |  |  | 502 |  |  | 1,532 |  |  |  |  |
| Net cash provided by operating activities | 14,747 |  |  | 14,923 |  |  | 13,256 |  |  |  |  |
| Cash Flows from Investing Activities |  |  |  |  |  |  |  |  |  |  |  |
| Purchases of property and equipment excluding finance leases, net of sales | ( 8,527 ) |  |  | ( 11,342 ) |  |  | ( 8,899 ) |  |  |  |  |
| Purchases of investments | ( 37,109 ) |  |  | ( 35,955 ) |  |  | ( 19,112 ) |  |  |  |  |
| Proceeds from maturities of investments | 30,158 |  |  | 28,310 |  |  | 12,353 |  |  |  |  |
| Proceeds from sales of investments | — |  |  | 200 |  |  | 138 |  |  |  |  |
| Business combinations, net of cash acquired | — |  |  | — |  |  | ( 64 ) |  |  |  |  |
| Net cash used in investing activities | ( 15,478 ) |  |  | ( 18,787 ) |  |  | ( 15,584 ) |  |  |  |  |
| Cash Flows from Financing Activities |  |  |  |  |  |  |  |  |  |  |  |
| Proceeds from issuances of debt | 5,586 |  |  | 5,744 |  |  | 3,931 |  |  |  |  |
| Repayments of debt | ( 5,546 ) |  |  | ( 2,500 ) |  |  | ( 1,351 ) |  |  |  |  |
| Net cash provided by financing activities | 1,139 |  |  | 3,853 |  |  | 2,589 |  |  |  |  |"""

AAPL_FY2025_INCOME_STATEMENT = """### Years ended
| Col_1 | September 27, 2025 | Col_3 | September 28, 2024 | Col_5 | September 30, 2023 | Col_7 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Net sales: |  |  |  |  |  |  |
| Products | $ | 307,003 | $ | 294,866 | $ | 298,085 |
| Services | 109,158 |  | 96,169 |  | 85,200 |  |
| Total net sales | 416,161 |  | 391,035 |  | 383,285 |  |
| Cost of sales: |  |  |  |  |  |  |
| Products | 186,300 |  | 185,233 |  | 189,282 |  |
| Services | 34,660 |  | 25,119 |  | 24,855 |  |
| Total cost of sales | 220,960 |  | 210,352 |  | 214,137 |  |
| Gross margin | 195,201 |  | 180,683 |  | 169,148 |  |
| Operating expenses: |  |  |  |  |  |  |
| Research and development | 34,550 |  | 31,370 |  | 29,915 |  |
| Selling, general and administrative | 27,601 |  | 26,097 |  | 24,932 |  |
| Total operating expenses | 62,151 |  | 57,467 |  | 54,847 |  |
| Operating income | 133,050 |  | 123,216 |  | 114,301 |  |
| Other income/(expense), net | ( 321 ) |  | 269 |  | ( 565 ) |  |
| Income before provision for income taxes | 132,729 |  | 123,485 |  | 113,736 |  |
| Provision for income taxes | 20,719 |  | 29,749 |  | 16,741 |  |
| Net income | $ | 112,010 | $ | 93,736 | $ | 96,995 |
| Earnings per share: |  |  |  |  |  |  |
| Basic | $ | 7.49 | $ | 6.11 | $ | 6.16 |
| Diluted | $ | 7.46 | $ | 6.08 | $ | 6.13 |
| Shares used in computing earnings per share: |  |  |  |  |  |  |
| Basic | 14,948,500 |  | 15,343,783 |  | 15,744,231 |  |
| Diluted | 15,004,697 |  | 15,408,095 |  | 15,812,547 |  |"""

AAPL_FY2025_BALANCE_SHEET = """| Col_1 | September 27, 2025 | Col_3 | September 28, 2024 | Col_5 |
| :--- | :--- | :--- | :--- | :--- |
| ASSETS: |  |  |  |  |
| Current assets: |  |  |  |  |
| Cash and cash equivalents | $ | 35,934 | $ | 29,943 |
| Marketable securities | 18,763 |  | 35,228 |  |
| Accounts receivable, net | 39,777 |  | 33,410 |  |
| Inventories | 5,718 |  | 7,286 |  |
| Other current assets | 47,765 |  | 47,120 |  |
| Total current assets | 147,957 |  | 152,987 |  |
| Non-current assets: |  |  |  |  |
| Marketable securities | 77,723 |  | 91,479 |  |
| Property, plant and equipment, net | 49,834 |  | 45,680 |  |
| Other non-current assets | 83,727 |  | 74,834 |  |
| Total non-current assets | 211,284 |  | 211,993 |  |
| Total assets | $ | 359,241 | $ | 364,980 |
| LIABILITIES AND SHAREHOLDERS’ EQUITY: |  |  |  |  |
| Current liabilities: |  |  |  |  |
| Accounts payable | $ | 69,860 | $ | 68,960 |
| Other current liabilities | 75,442 |  | 78,304 |  |
| Commercial paper | 7,979 |  | 9,967 |  |
| Term debt | 12,350 |  | 10,912 |  |
| Total current liabilities | 165,631 |  | 176,392 |  |
| Non-current liabilities: |  |  |  |  |
| Term debt | 78,328 |  | 85,750 |  |
| Other non-current liabilities | 41,549 |  | 45,888 |  |
| Total non-current liabilities | 119,877 |  | 131,638 |  |
| Total liabilities | 285,508 |  | 308,030 |  |
| Shareholders’ equity: |  |  |  |  |
| Common stock and additional paid-in capital, $0.00001 par value: 50,400,000 shares authorized; 14,773,260 and 15,116,786 shares issued and outstanding, respectively | 93,568 |  | 83,276 |  |
| Accumulated deficit | ( 14,264 ) |  | ( 19,154 ) |  |
| Accumulated other comprehensive loss | ( 5,571 ) |  | ( 7,172 ) |  |
| Total shareholders’ equity | 73,733 |  | 56,950 |  |
| Total liabilities and shareholders’ equity | $ | 359,241 | $ | 364,980 |"""

AAPL_FY2025_CASH_FLOW = """### Years ended
| Col_1 | September 27, 2025 | Col_3 | September 28, 2024 | Col_5 | September 30, 2023 | Col_7 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Operating activities: |  |  |  |  |  |  |
| Net income | $ | 112,010 | $ | 93,736 | $ | 96,995 |
| Adjustments to reconcile net income to cash generated by operating activities: |  |  |  |  |  |  |
| Depreciation and amortization | 11,698 |  | 11,445 |  | 11,519 |  |
| Share-based compensation expense | 12,863 |  | 11,688 |  | 10,833 |  |
| Other | ( 89 ) |  | ( 2,266 ) |  | ( 2,227 ) |  |
| Changes in operating assets and liabilities: |  |  |  |  |  |  |
| Accounts receivable, net | ( 6,682 ) |  | ( 3,788 ) |  | ( 1,688 ) |  |
| Other current and non-current liabilities | ( 18,318 ) |  | 15,552 |  | 3,031 |  |
| Cash generated by operating activities | 111,482 |  | 118,254 |  | 110,543 |  |
| Investing activities: |  |  |  |  |  |  |
| Purchases of marketable securities | ( 29,513 ) |  | ( 48,656 ) |  | ( 29,513 ) |  |
| Payments for acquisition of property, plant and equipment | ( 12,715 ) |  | ( 9,447 ) |  | ( 10,959 ) |  |
| Cash generated by/(used in) investing activities | 15,195 |  | 2,935 |  | 3,705 |  |"""


def tsla_payloads(filing_year: int = 2025):
    """Retrieval-tool style payloads (newest filing first) for the TSLA FY2025 filing."""
    return {
        "income_statement": [{"chunk_id": "tsla-is", "fiscal_year": filing_year, "item": "Item 8",
                              "breadcrumb": "Tesla, Inc. (TSLA) > 10-K FY2025 > Item 8",
                              "sub_section": "Year Ended December 31,",
                              "statement_type": "income_statement",
                              "table_markdown": TSLA_FY2025_INCOME_STATEMENT}],
        "balance_sheet": [{"chunk_id": "tsla-bs", "fiscal_year": filing_year, "item": "Item 8",
                           "breadcrumb": "Tesla, Inc. (TSLA) > 10-K FY2025 > Item 8",
                           "sub_section": "Consolidated Balance Sheets",
                           "statement_type": "balance_sheet",
                           "table_markdown": TSLA_FY2025_BALANCE_SHEET}],
        "cash_flow": [{"chunk_id": "tsla-cf", "fiscal_year": filing_year, "item": "Item 8",
                       "breadcrumb": "Tesla, Inc. (TSLA) > 10-K FY2025 > Item 8",
                       "sub_section": "Year Ended December 31,",
                       "statement_type": "cash_flow",
                       "table_markdown": TSLA_FY2025_CASH_FLOW}],
    }


def aapl_payloads(filing_year: int = 2025):
    return {
        "income_statement": [{"chunk_id": "aapl-is", "fiscal_year": filing_year, "statement_type": "income_statement",
                              "table_markdown": AAPL_FY2025_INCOME_STATEMENT}],
        "balance_sheet": [{"chunk_id": "aapl-bs", "fiscal_year": filing_year, "statement_type": "balance_sheet",
                           "table_markdown": AAPL_FY2025_BALANCE_SHEET}],
        "cash_flow": [{"chunk_id": "aapl-cf", "fiscal_year": filing_year, "statement_type": "cash_flow",
                       "table_markdown": AAPL_FY2025_CASH_FLOW}],
    }
