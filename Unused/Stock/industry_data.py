"""
Exhaustive Industry Classification Data

Granular sub-industry classification for US equities, covering 100+ sub-industries
across 12 sectors. Used for relative strength analysis and peer comparison.

Standard GICS sectors (11 sectors) are too broad for meaningful relative strength
comparison. This module provides granular sub-industries.
"""

from typing import Dict, List, Optional

# =============================================================================
# INDUSTRY CLASSIFICATION
# =============================================================================

INDUSTRY_CLASSIFICATION: Dict[str, Dict[str, Dict]] = {
    # =========================================================================
    # TECHNOLOGY
    # =========================================================================
    'technology': {
        'artificial_intelligence': {
            'description': 'AI/ML infrastructure, models, and applications',
            'examples': ['NVDA', 'AMD', 'GOOGL', 'MSFT', 'PLTR', 'AI', 'PATH', 'SNOW', 'CRM', 'ADBE'],
            'keywords': ['artificial intelligence', 'machine learning', 'neural network', 'LLM', 'GPU', 'AI'],
        },
        'quantum_computing': {
            'description': 'Quantum hardware, software, and services',
            'examples': ['IONQ', 'RGTI', 'QBTS', 'IBM', 'GOOGL', 'MSFT'],
            'keywords': ['quantum', 'qubit', 'quantum computing', 'quantum supremacy'],
        },
        'semiconductors_logic': {
            'description': 'CPU, GPU, FPGA, ASIC designers',
            'examples': ['NVDA', 'AMD', 'INTC', 'QCOM', 'AVGO', 'MRVL', 'ARM', 'ADI', 'TXN', 'NXPI'],
            'keywords': ['semiconductor', 'chip', 'processor', 'GPU', 'CPU', 'FPGA', 'ASIC'],
        },
        'semiconductors_memory': {
            'description': 'DRAM, NAND, storage chips',
            'examples': ['MU', 'WDC', 'STX', 'SSNLF', 'KIOXIA'],
            'keywords': ['memory', 'DRAM', 'NAND', 'flash', 'storage', 'SSD'],
        },
        'semiconductor_equipment': {
            'description': 'Chip manufacturing equipment',
            'examples': ['ASML', 'LRCX', 'AMAT', 'KLAC', 'TER', 'ONTO', 'ACLS', 'MKSI'],
            'keywords': ['lithography', 'wafer', 'fab equipment', 'etch', 'deposition', 'EUV'],
        },
        'cloud_infrastructure': {
            'description': 'Cloud platforms, IaaS, PaaS',
            'examples': ['AMZN', 'MSFT', 'GOOGL', 'ORCL', 'IBM', 'DDOG', 'NET', 'FSLY', 'MDB'],
            'keywords': ['cloud', 'AWS', 'Azure', 'GCP', 'data center', 'IaaS', 'PaaS'],
        },
        'enterprise_software': {
            'description': 'B2B software, SaaS, productivity',
            'examples': ['CRM', 'NOW', 'WDAY', 'TEAM', 'ZM', 'DOCU', 'ADBE', 'INTU', 'CDNS', 'SNPS'],
            'keywords': ['SaaS', 'enterprise', 'B2B', 'software', 'ERP', 'CRM'],
        },
        'cybersecurity': {
            'description': 'Security software, services, hardware',
            'examples': ['CRWD', 'PANW', 'ZS', 'FTNT', 'S', 'OKTA', 'CYBR', 'NET', 'TENB', 'RPD'],
            'keywords': ['security', 'cyber', 'firewall', 'threat', 'endpoint', 'SIEM', 'zero trust'],
        },
        'fintech': {
            'description': 'Financial technology, payments, neobanks',
            'examples': ['SQ', 'PYPL', 'AFRM', 'SOFI', 'UPST', 'NU', 'COIN', 'HOOD', 'BILL', 'TOST'],
            'keywords': ['fintech', 'payment', 'digital banking', 'BNPL', 'neobank', 'crypto'],
        },
        'e_commerce_platforms': {
            'description': 'Online retail platforms, marketplaces',
            'examples': ['AMZN', 'SHOP', 'ETSY', 'EBAY', 'MELI', 'SE', 'JD', 'BABA', 'PDD', 'CPNG'],
            'keywords': ['e-commerce', 'marketplace', 'online retail', 'shopping'],
        },
        'social_media': {
            'description': 'Social networks, content platforms',
            'examples': ['META', 'SNAP', 'PINS', 'RDDT', 'MTCH', 'BMBL'],
            'keywords': ['social media', 'social network', 'content', 'advertising', 'dating'],
        },
        'gaming_esports': {
            'description': 'Video games, esports, interactive entertainment',
            'examples': ['EA', 'TTWO', 'RBLX', 'U', 'ATVI', 'NTDOY', 'SONY', 'SKLZ', 'HUYA'],
            'keywords': ['gaming', 'video game', 'esports', 'metaverse', 'mobile games'],
        },
        'streaming_media': {
            'description': 'Video/audio streaming services',
            'examples': ['NFLX', 'DIS', 'SPOT', 'ROKU', 'WBD', 'PARA', 'FUBO', 'AMC'],
            'keywords': ['streaming', 'OTT', 'video on demand', 'music streaming'],
        },
        'adtech_martech': {
            'description': 'Advertising technology, marketing platforms',
            'examples': ['TTD', 'DV', 'APPS', 'PUBM', 'MGNI', 'CARG', 'CRTO', 'ZETA'],
            'keywords': ['advertising', 'programmatic', 'digital marketing', 'adtech', 'DSP'],
        },
        'data_analytics': {
            'description': 'Big data, analytics, business intelligence',
            'examples': ['SNOW', 'MDB', 'DDOG', 'SPLK', 'ESTC', 'PLAN', 'AYX', 'CLDR', 'PLTR'],
            'keywords': ['analytics', 'big data', 'data warehouse', 'BI', 'observability'],
        },
        'it_services_consulting': {
            'description': 'IT consulting, system integrators',
            'examples': ['ACN', 'IBM', 'INFY', 'WIT', 'CTSH', 'EPAM', 'GLOB', 'DXC', 'GIB'],
            'keywords': ['IT services', 'consulting', 'outsourcing', 'system integration'],
        },
        'hardware_networking': {
            'description': 'Network equipment, hardware',
            'examples': ['CSCO', 'ANET', 'JNPR', 'HPE', 'DELL', 'NTAP', 'PSTG', 'INFN', 'SMCI'],
            'keywords': ['networking', 'switch', 'router', 'hardware', 'server', 'storage'],
        },
        'consumer_electronics': {
            'description': 'Phones, computers, wearables',
            'examples': ['AAPL', 'SONY', 'LOGI', 'SONO', 'GPRO', 'KOSS', 'HEAR'],
            'keywords': ['consumer electronics', 'smartphone', 'wearable', 'headphones'],
        },
    },

    # =========================================================================
    # HEALTHCARE & BIOTECH
    # =========================================================================
    'healthcare': {
        'biotech_large_cap': {
            'description': 'Large biotech with revenue',
            'examples': ['AMGN', 'GILD', 'BIIB', 'REGN', 'VRTX', 'MRNA', 'BNTX', 'ALNY', 'SGEN'],
            'keywords': ['biotech', 'biopharmaceutical', 'biotechnology'],
        },
        'biotech_clinical_stage': {
            'description': 'Clinical-stage drug developers',
            'examples': ['SAVA', 'SRPT', 'RCUS', 'IMVT', 'AKRO', 'KYMR', 'RAPT', 'TVTX', 'VKTX'],
            'keywords': ['clinical trial', 'Phase 2', 'Phase 3', 'FDA', 'pipeline'],
        },
        'biotech_oncology': {
            'description': 'Cancer-focused biotech',
            'examples': ['SGEN', 'EXEL', 'MGNX', 'FATE', 'BCYC', 'MRTX', 'ARVN', 'ERAS', 'PCVX'],
            'keywords': ['oncology', 'cancer', 'tumor', 'immunotherapy', 'CAR-T'],
        },
        'biotech_rare_disease': {
            'description': 'Orphan drugs, rare disease',
            'examples': ['VRTX', 'ALNY', 'BMRN', 'SRPT', 'RARE', 'FOLD', 'IONS', 'NBIX'],
            'keywords': ['rare disease', 'orphan drug', 'genetic', 'gene therapy'],
        },
        'biotech_gene_therapy': {
            'description': 'Gene/cell therapy developers',
            'examples': ['CRSP', 'EDIT', 'NTLA', 'BEAM', 'BLUE', 'SGMO', 'VRTX', 'ABCL'],
            'keywords': ['gene therapy', 'CRISPR', 'cell therapy', 'CAR-T', 'gene editing'],
        },
        'biotech_neurology': {
            'description': 'Neurology-focused biotech',
            'examples': ['BIIB', 'SAVA', 'PRAX', 'ANNX', 'CERE', 'AXSM', 'SAGE', 'NBIX'],
            'keywords': ['neurology', 'Alzheimer', 'Parkinson', 'CNS', 'brain'],
        },
        'pharmaceuticals_major': {
            'description': 'Big pharma, diversified',
            'examples': ['JNJ', 'PFE', 'MRK', 'LLY', 'ABBV', 'BMY', 'NVS', 'AZN', 'GSK', 'SNY'],
            'keywords': ['pharmaceutical', 'pharma', 'drug', 'big pharma'],
        },
        'pharmaceuticals_specialty': {
            'description': 'Specialty pharma, branded generics',
            'examples': ['JAZZ', 'NBIX', 'UTHR', 'INCY', 'HZNP', 'BHVN', 'PTCT', 'IRWD'],
            'keywords': ['specialty pharma', 'branded', 'specialty drug'],
        },
        'medical_devices': {
            'description': 'Medical equipment, devices',
            'examples': ['MDT', 'ABT', 'BSX', 'SYK', 'EW', 'ISRG', 'DXCM', 'HOLX', 'ZBH', 'BDX'],
            'keywords': ['medical device', 'implant', 'surgical', 'diagnostic', 'robotic surgery'],
        },
        'healthcare_services': {
            'description': 'Hospitals, clinics, healthcare providers',
            'examples': ['UNH', 'HCA', 'THC', 'DVA', 'ACHC', 'CYH', 'EHC', 'SEM', 'ENSG'],
            'keywords': ['hospital', 'healthcare provider', 'clinic', 'dialysis'],
        },
        'health_insurance': {
            'description': 'Health insurers, managed care',
            'examples': ['UNH', 'ELV', 'CI', 'HUM', 'CNC', 'MOH', 'CVS'],
            'keywords': ['health insurance', 'managed care', 'Medicare', 'Medicaid', 'PBM'],
        },
        'diagnostics_testing': {
            'description': 'Lab testing, diagnostics',
            'examples': ['DGX', 'LH', 'ILMN', 'TMO', 'EXAS', 'GH', 'NTRA', 'CDNA', 'NVTA'],
            'keywords': ['diagnostic', 'lab testing', 'genomics', 'sequencing', 'liquid biopsy'],
        },
        'healthcare_it': {
            'description': 'Healthcare software, EMR',
            'examples': ['VEEV', 'CERN', 'HIMS', 'TDOC', 'DOCS', 'AMWL', 'PHIC', 'HLTH'],
            'keywords': ['healthcare IT', 'EMR', 'telehealth', 'EHR', 'telemedicine'],
        },
        'life_science_tools': {
            'description': 'Lab equipment, research tools',
            'examples': ['TMO', 'DHR', 'A', 'PKI', 'BIO', 'TECH', 'MTD', 'WAT', 'ILMN'],
            'keywords': ['life science', 'lab equipment', 'research tools', 'instruments'],
        },
        'cannabis': {
            'description': 'Cannabis producers, dispensaries',
            'examples': ['TLRY', 'CGC', 'CRON', 'ACB', 'CURLF', 'GTBIF', 'TCNNF', 'CRLBF'],
            'keywords': ['cannabis', 'marijuana', 'CBD', 'THC', 'dispensary'],
        },
    },

    # =========================================================================
    # FINANCIALS
    # =========================================================================
    'financials': {
        'banks_money_center': {
            'description': 'Large money center banks',
            'examples': ['JPM', 'BAC', 'WFC', 'C', 'USB', 'PNC', 'TFC', 'GS', 'MS'],
            'keywords': ['bank', 'money center', 'commercial bank', 'universal bank'],
        },
        'banks_regional': {
            'description': 'Regional and community banks',
            'examples': ['FITB', 'KEY', 'RF', 'HBAN', 'CFG', 'ZION', 'MTB', 'CMA', 'FHN', 'EWBC'],
            'keywords': ['regional bank', 'community bank', 'super-regional'],
        },
        'investment_banks': {
            'description': 'Investment banking, capital markets',
            'examples': ['GS', 'MS', 'SCHW', 'RJF', 'EVR', 'LAZ', 'PJT', 'MC', 'HLI'],
            'keywords': ['investment bank', 'M&A', 'capital markets', 'advisory'],
        },
        'asset_managers': {
            'description': 'Asset management, mutual funds',
            'examples': ['BLK', 'BX', 'KKR', 'APO', 'ARES', 'OWL', 'CG', 'TROW', 'IVZ', 'BEN'],
            'keywords': ['asset management', 'fund', 'AUM', 'private equity', 'hedge fund'],
        },
        'insurance_life': {
            'description': 'Life insurance companies',
            'examples': ['MET', 'PRU', 'AFL', 'LNC', 'PFG', 'VOYA', 'UNM', 'GL', 'CNO'],
            'keywords': ['life insurance', 'annuity', 'retirement'],
        },
        'insurance_property': {
            'description': 'Property & casualty insurance',
            'examples': ['AIG', 'TRV', 'ALL', 'PGR', 'CB', 'HIG', 'CNA', 'AFG', 'CINF'],
            'keywords': ['P&C', 'property insurance', 'casualty', 'auto insurance'],
        },
        'insurance_specialty': {
            'description': 'Specialty insurance, reinsurance',
            'examples': ['RNR', 'ACGL', 'WRB', 'KNSL', 'RLI', 'PRI', 'RYAN', 'BRO'],
            'keywords': ['specialty insurance', 'reinsurance', 'E&S'],
        },
        'reits_residential': {
            'description': 'Apartment, single-family REITs',
            'examples': ['EQR', 'AVB', 'ESS', 'MAA', 'UDR', 'INVH', 'AMH', 'CPT', 'AIV'],
            'keywords': ['residential REIT', 'apartment', 'multifamily', 'single-family'],
        },
        'reits_commercial': {
            'description': 'Office, retail REITs',
            'examples': ['BXP', 'SLG', 'VNO', 'SPG', 'O', 'NNN', 'KIM', 'REG', 'FRT'],
            'keywords': ['commercial REIT', 'office', 'retail', 'mall', 'shopping center'],
        },
        'reits_industrial': {
            'description': 'Warehouse, logistics REITs',
            'examples': ['PLD', 'DRE', 'REXR', 'FR', 'STAG', 'EGP', 'COLD', 'TRNO'],
            'keywords': ['industrial REIT', 'warehouse', 'logistics', 'distribution'],
        },
        'reits_data_center': {
            'description': 'Data center REITs',
            'examples': ['EQIX', 'DLR', 'AMT', 'CCI', 'SBAC', 'UNIT', 'QTS'],
            'keywords': ['data center REIT', 'tower REIT', 'colocation', 'cell tower'],
        },
        'reits_healthcare': {
            'description': 'Healthcare property REITs',
            'examples': ['WELL', 'VTR', 'PEAK', 'OHI', 'HR', 'CTRE', 'DOC', 'LTC'],
            'keywords': ['healthcare REIT', 'senior housing', 'medical office', 'nursing'],
        },
        'mortgage_finance': {
            'description': 'Mortgage REITs, mortgage servicers',
            'examples': ['AGNC', 'NLY', 'RKT', 'UWMC', 'COOP', 'PFSI', 'GHLD', 'TWO'],
            'keywords': ['mortgage', 'mREIT', 'lending', 'mortgage servicer'],
        },
        'exchanges_brokers': {
            'description': 'Exchanges, trading platforms, brokers',
            'examples': ['ICE', 'CME', 'NDAQ', 'CBOE', 'IBKR', 'HOOD', 'VIRT', 'MKTX'],
            'keywords': ['exchange', 'trading', 'broker', 'clearinghouse', 'market maker'],
        },
        'crypto_blockchain': {
            'description': 'Crypto exchanges, miners, blockchain',
            'examples': ['COIN', 'MARA', 'RIOT', 'MSTR', 'HUT', 'CLSK', 'BTBT', 'CIFR', 'IREN'],
            'keywords': ['crypto', 'bitcoin', 'blockchain', 'mining', 'cryptocurrency'],
        },
    },

    # =========================================================================
    # ENERGY
    # =========================================================================
    'energy': {
        'oil_gas_majors': {
            'description': 'Integrated oil & gas majors',
            'examples': ['XOM', 'CVX', 'SHEL', 'BP', 'TTE', 'COP', 'ENB'],
            'keywords': ['integrated oil', 'major', 'supermajor', 'IOC'],
        },
        'oil_gas_exploration': {
            'description': 'E&P, upstream oil & gas',
            'examples': ['EOG', 'PXD', 'DVN', 'FANG', 'OXY', 'COP', 'MRO', 'APA', 'MTDR', 'PR'],
            'keywords': ['E&P', 'exploration', 'upstream', 'shale', 'Permian', 'unconventional'],
        },
        'oil_gas_services': {
            'description': 'Oilfield services, drilling',
            'examples': ['SLB', 'HAL', 'BKR', 'FTI', 'NOV', 'CHX', 'WHD', 'HP', 'PTEN'],
            'keywords': ['oilfield services', 'drilling', 'completion', 'well services'],
        },
        'oil_refining': {
            'description': 'Refiners, downstream',
            'examples': ['VLO', 'MPC', 'PSX', 'DK', 'PBF', 'HFC', 'PARR', 'CVI'],
            'keywords': ['refining', 'refiner', 'downstream', 'crack spread', 'fuels'],
        },
        'natural_gas': {
            'description': 'Natural gas producers, midstream',
            'examples': ['EQT', 'RRC', 'AR', 'SWN', 'CNX', 'CHK', 'CTRA', 'GPOR'],
            'keywords': ['natural gas', 'LNG', 'gas producer', 'Marcellus', 'Appalachian'],
        },
        'midstream_pipelines': {
            'description': 'Pipelines, MLPs, midstream',
            'examples': ['WMB', 'KMI', 'OKE', 'EPD', 'ET', 'MPLX', 'TRGP', 'PAA', 'HESM'],
            'keywords': ['pipeline', 'midstream', 'MLP', 'gathering', 'processing'],
        },
        'lng_export': {
            'description': 'LNG export terminals, liquefaction',
            'examples': ['LNG', 'TELL', 'GLNG', 'CQP', 'NFE', 'NEXT'],
            'keywords': ['LNG', 'liquefaction', 'export terminal', 'natural gas liquids'],
        },
        'uranium_nuclear': {
            'description': 'Uranium miners, nuclear fuel',
            'examples': ['CCJ', 'UEC', 'UUUU', 'DNN', 'NXE', 'LEU', 'URG', 'SRUUF'],
            'keywords': ['uranium', 'nuclear', 'enrichment', 'nuclear fuel', 'yellowcake'],
        },
        'coal': {
            'description': 'Coal mining, met coal',
            'examples': ['BTU', 'ARCH', 'AMR', 'CEIX', 'HCC', 'ARLP', 'METC'],
            'keywords': ['coal', 'metallurgical coal', 'thermal coal', 'met coal'],
        },
    },

    # =========================================================================
    # CLEAN ENERGY & RENEWABLES
    # =========================================================================
    'clean_energy': {
        'solar': {
            'description': 'Solar panels, inverters, installers',
            'examples': ['ENPH', 'SEDG', 'FSLR', 'RUN', 'NOVA', 'ARRY', 'MAXN', 'JKS', 'CSIQ'],
            'keywords': ['solar', 'photovoltaic', 'PV', 'solar panel', 'inverter', 'residential solar'],
        },
        'wind': {
            'description': 'Wind turbines, wind farms',
            'examples': ['TPIC', 'GE', 'VWDRY', 'BWXT', 'ORA'],
            'keywords': ['wind', 'turbine', 'offshore wind', 'wind farm', 'renewable'],
        },
        'energy_storage': {
            'description': 'Batteries, grid storage',
            'examples': ['STEM', 'FLUX', 'FLNC', 'ENVX', 'QS', 'FREYR', 'MVST', 'DCFC'],
            'keywords': ['battery', 'energy storage', 'grid storage', 'ESS', 'solid state'],
        },
        'hydrogen_fuel_cells': {
            'description': 'Hydrogen production, fuel cells',
            'examples': ['PLUG', 'BE', 'FCEL', 'BLDP', 'HYSR', 'NKLA', 'HTOO'],
            'keywords': ['hydrogen', 'fuel cell', 'electrolyzer', 'green hydrogen', 'H2'],
        },
        'ev_manufacturers': {
            'description': 'Electric vehicle makers',
            'examples': ['TSLA', 'RIVN', 'LCID', 'NIO', 'XPEV', 'LI', 'FSR', 'VFS', 'PSNY'],
            'keywords': ['electric vehicle', 'EV', 'BEV', 'electric car', 'Tesla'],
        },
        'ev_charging': {
            'description': 'EV charging networks, equipment',
            'examples': ['CHPT', 'BLNK', 'EVGO', 'DCFC', 'AMPX', 'PTRA'],
            'keywords': ['EV charging', 'charger', 'charging station', 'DCFC', 'fast charging'],
        },
        'ev_components': {
            'description': 'EV batteries, motors, components',
            'examples': ['LAC', 'ALB', 'LTHM', 'MP', 'PCRFY', 'SQM', 'PLL', 'SGML'],
            'keywords': ['EV battery', 'lithium', 'rare earth', 'cathode', 'battery materials'],
        },
        'clean_utilities': {
            'description': 'Renewable-focused utilities',
            'examples': ['NEE', 'AES', 'ORA', 'CWEN', 'BEP', 'BEPC', 'RNW'],
            'keywords': ['renewable utility', 'clean energy', 'yieldco', 'IPP'],
        },
        'carbon_capture': {
            'description': 'Carbon capture, sequestration',
            'examples': ['ACES', 'OXY', 'XOM', 'CVX'],
            'keywords': ['carbon capture', 'CCS', 'CCUS', 'sequestration', 'DAC'],
        },
    },

    # =========================================================================
    # INDUSTRIALS
    # =========================================================================
    'industrials': {
        'aerospace_defense': {
            'description': 'Defense contractors, aerospace',
            'examples': ['LMT', 'RTX', 'NOC', 'GD', 'BA', 'LHX', 'HII', 'TDG', 'HEI', 'AXON'],
            'keywords': ['defense', 'aerospace', 'military', 'contractor', 'DOD'],
        },
        'airlines': {
            'description': 'Passenger airlines',
            'examples': ['DAL', 'UAL', 'AAL', 'LUV', 'JBLU', 'ALK', 'SAVE', 'HA', 'SKYW'],
            'keywords': ['airline', 'aviation', 'passenger', 'carrier', 'travel'],
        },
        'railroads': {
            'description': 'Freight railroads',
            'examples': ['UNP', 'CSX', 'NSC', 'CP', 'CNI', 'KSU'],
            'keywords': ['railroad', 'rail', 'freight rail', 'intermodal', 'Class I'],
        },
        'trucking_logistics': {
            'description': 'Trucking, logistics, freight',
            'examples': ['UPS', 'FDX', 'XPO', 'ODFL', 'JBHT', 'CHRW', 'SAIA', 'KNX', 'WERN'],
            'keywords': ['trucking', 'logistics', 'freight', 'delivery', 'LTL', 'truckload'],
        },
        'shipping_marine': {
            'description': 'Container shipping, tankers, bulk',
            'examples': ['ZIM', 'MATX', 'DAC', 'INSW', 'STNG', 'GOGL', 'EGLE', 'NMM', 'TRMD'],
            'keywords': ['shipping', 'container', 'tanker', 'dry bulk', 'maritime', 'LNG carrier'],
        },
        'machinery_equipment': {
            'description': 'Industrial machinery, equipment',
            'examples': ['CAT', 'DE', 'PCAR', 'CMI', 'IR', 'PH', 'ITW', 'EMR', 'ROK'],
            'keywords': ['machinery', 'equipment', 'industrial', 'construction equipment'],
        },
        'construction_engineering': {
            'description': 'Construction, engineering services',
            'examples': ['JCI', 'EME', 'PWR', 'FLR', 'J', 'MTZ', 'STRL', 'DY', 'GVA'],
            'keywords': ['construction', 'engineering', 'EPC', 'infrastructure', 'contractor'],
        },
        'building_materials': {
            'description': 'Cement, aggregates, building products',
            'examples': ['VMC', 'MLM', 'CX', 'SUM', 'EXP', 'BLDR', 'BLD', 'USLM', 'FRTA'],
            'keywords': ['building materials', 'cement', 'aggregates', 'concrete', 'asphalt'],
        },
        'homebuilders': {
            'description': 'Residential homebuilders',
            'examples': ['DHI', 'LEN', 'NVR', 'PHM', 'TOL', 'KBH', 'MHO', 'MTH', 'TMHC', 'CCS'],
            'keywords': ['homebuilder', 'residential', 'housing', 'new homes', 'construction'],
        },
        'waste_management': {
            'description': 'Waste collection, recycling',
            'examples': ['WM', 'RSG', 'WCN', 'CLH', 'CWST', 'GFL', 'SRCL', 'ATKR'],
            'keywords': ['waste', 'recycling', 'landfill', 'environmental services'],
        },
        'electrical_equipment': {
            'description': 'Electrical components, equipment',
            'examples': ['ETN', 'EMR', 'ROK', 'AME', 'GNRC', 'HUBB', 'AYI', 'POWL'],
            'keywords': ['electrical equipment', 'automation', 'power', 'electrical components'],
        },
        'tools_hardware': {
            'description': 'Hand tools, power tools, hardware',
            'examples': ['SWK', 'FAST', 'GWW', 'MSM', 'HD', 'LOW', 'SITE', 'POOL'],
            'keywords': ['tools', 'hardware', 'fasteners', 'MRO', 'distribution'],
        },
        'staffing_hr': {
            'description': 'Staffing, HR services',
            'examples': ['RHI', 'ASGN', 'KFRC', 'MAN', 'HEES', 'NSP', 'PAYX', 'ADP'],
            'keywords': ['staffing', 'HR', 'employment', 'recruiting', 'payroll'],
        },
        'security_services': {
            'description': 'Physical security, monitoring',
            'examples': ['ALLE', 'SRCL', 'AOS', 'SCI', 'BCO', 'NSSC'],
            'keywords': ['security', 'monitoring', 'alarm', 'security services'],
        },
        'robotics_automation': {
            'description': 'Industrial robotics, automation',
            'examples': ['ROK', 'TER', 'ISRG', 'CGNX', 'BRKS', 'KTOS', 'IRBT', 'NOVT'],
            'keywords': ['robotics', 'automation', 'industrial robot', 'cobot'],
        },
        '3d_printing': {
            'description': 'Additive manufacturing',
            'examples': ['DDD', 'SSYS', 'XONE', 'DM', 'MTLS', 'NNDM', 'VELO'],
            'keywords': ['3D printing', 'additive manufacturing', 'rapid prototyping'],
        },
        'drones_uav': {
            'description': 'Drones, unmanned systems',
            'examples': ['AVAV', 'KTOS', 'JOBY', 'ACHR', 'LUNR', 'RKLB', 'EVTL'],
            'keywords': ['drone', 'UAV', 'eVTOL', 'unmanned', 'air taxi'],
        },
        'space': {
            'description': 'Space technology, satellites',
            'examples': ['RKLB', 'SPCE', 'ASTS', 'BKSY', 'RDW', 'LUNR', 'PL', 'MNTS', 'ASTR'],
            'keywords': ['space', 'satellite', 'rocket', 'launch', 'orbital', 'LEO'],
        },
    },

    # =========================================================================
    # CONSUMER DISCRETIONARY
    # =========================================================================
    'consumer_discretionary': {
        'auto_manufacturers': {
            'description': 'Traditional auto OEMs',
            'examples': ['F', 'GM', 'TM', 'HMC', 'STLA', 'VWAGY', 'MBGYY', 'BMW'],
            'keywords': ['auto', 'automotive', 'car manufacturer', 'OEM', 'vehicle'],
        },
        'auto_parts': {
            'description': 'Auto parts suppliers',
            'examples': ['APTV', 'BWA', 'LEA', 'ADNT', 'MGA', 'ALV', 'AXL', 'VC', 'DAN'],
            'keywords': ['auto parts', 'supplier', 'OEM supplier', 'tier 1'],
        },
        'restaurants_fast_food': {
            'description': 'Quick service restaurants',
            'examples': ['MCD', 'SBUX', 'YUM', 'QSR', 'CMG', 'WEN', 'DPZ', 'JACK', 'WING'],
            'keywords': ['fast food', 'QSR', 'restaurant', 'drive-thru', 'franchise'],
        },
        'restaurants_casual': {
            'description': 'Casual dining restaurants',
            'examples': ['DRI', 'TXRH', 'EAT', 'BLMN', 'CAKE', 'CBRL', 'RUTH', 'BJRI'],
            'keywords': ['casual dining', 'restaurant', 'sit-down', 'family dining'],
        },
        'hotels_resorts': {
            'description': 'Hotels, resorts, lodging',
            'examples': ['MAR', 'HLT', 'H', 'IHG', 'WH', 'CHH', 'PLYA', 'STAY', 'APTS'],
            'keywords': ['hotel', 'resort', 'lodging', 'hospitality', 'timeshare'],
        },
        'cruise_lines': {
            'description': 'Cruise operators',
            'examples': ['CCL', 'RCL', 'NCLH', 'LVS', 'VIK'],
            'keywords': ['cruise', 'cruise line', 'ocean cruise', 'river cruise'],
        },
        'casinos_gaming': {
            'description': 'Casinos, gaming, sports betting',
            'examples': ['LVS', 'WYNN', 'MGM', 'CZR', 'DKNG', 'PENN', 'FLUT', 'RSI', 'GENI'],
            'keywords': ['casino', 'gaming', 'gambling', 'sports betting', 'iGaming'],
        },
        'apparel_fashion': {
            'description': 'Apparel brands, fashion',
            'examples': ['NKE', 'LULU', 'VFC', 'PVH', 'RL', 'GOOS', 'ONON', 'DECK', 'SKX', 'UAA'],
            'keywords': ['apparel', 'fashion', 'clothing', 'sportswear', 'footwear'],
        },
        'luxury_goods': {
            'description': 'Luxury brands, premium goods',
            'examples': ['LVMHF', 'RMS', 'PPRUF', 'TPR', 'CPRI', 'BURBY', 'MONC'],
            'keywords': ['luxury', 'premium', 'high-end', 'designer', 'fashion house'],
        },
        'retail_specialty': {
            'description': 'Specialty retail',
            'examples': ['TJX', 'ROST', 'ULTA', 'FIVE', 'DG', 'DLTR', 'BBY', 'AAP', 'ORLY'],
            'keywords': ['specialty retail', 'discount', 'off-price', 'dollar store'],
        },
        'retail_home': {
            'description': 'Home improvement, furnishings',
            'examples': ['HD', 'LOW', 'RH', 'WSM', 'W', 'ETSY', 'ARHS', 'LL', 'FND'],
            'keywords': ['home improvement', 'furniture', 'home goods', 'decor'],
        },
        'leisure_recreation': {
            'description': 'Leisure, fitness, recreation',
            'examples': ['PTON', 'PLNT', 'POOL', 'BC', 'HOG', 'YETI', 'JAKK', 'PRTY'],
            'keywords': ['fitness', 'recreation', 'leisure', 'outdoor', 'sports equipment'],
        },
        'theme_parks': {
            'description': 'Theme parks, entertainment venues',
            'examples': ['DIS', 'SIX', 'FUN', 'SEAS', 'CMCSA'],
            'keywords': ['theme park', 'amusement', 'entertainment', 'attractions'],
        },
        'live_events': {
            'description': 'Concerts, events, ticketing',
            'examples': ['LYV', 'MSG', 'MSGE', 'SPHR', 'EDR', 'TKO'],
            'keywords': ['concert', 'live entertainment', 'ticketing', 'events', 'sports'],
        },
    },

    # =========================================================================
    # CONSUMER STAPLES
    # =========================================================================
    'consumer_staples': {
        'beverages_alcoholic': {
            'description': 'Beer, wine, spirits',
            'examples': ['BUD', 'TAP', 'STZ', 'DEO', 'BF.B', 'SAM', 'ABEV', 'MGPI'],
            'keywords': ['beer', 'wine', 'spirits', 'alcohol', 'brewing', 'distilling'],
        },
        'beverages_non_alcoholic': {
            'description': 'Soft drinks, energy drinks',
            'examples': ['KO', 'PEP', 'MNST', 'CELH', 'KDP', 'FIZZ', 'COCO'],
            'keywords': ['soft drink', 'beverage', 'energy drink', 'soda', 'water'],
        },
        'food_packaged': {
            'description': 'Packaged foods, snacks',
            'examples': ['GIS', 'K', 'MDLZ', 'HSY', 'SJM', 'CAG', 'CPB', 'POST', 'BGS', 'THS'],
            'keywords': ['packaged food', 'snack', 'cereal', 'confectionery', 'baked goods'],
        },
        'food_meat': {
            'description': 'Meat, poultry processing',
            'examples': ['TSN', 'HRL', 'PPC', 'SAFM', 'JBS', 'BRFS'],
            'keywords': ['meat', 'poultry', 'protein', 'beef', 'pork', 'chicken'],
        },
        'grocery_retail': {
            'description': 'Supermarkets, grocery',
            'examples': ['KR', 'ACI', 'SFM', 'CASY', 'GO', 'WMT', 'TGT', 'COST'],
            'keywords': ['grocery', 'supermarket', 'food retail', 'convenience'],
        },
        'household_products': {
            'description': 'Cleaning, household products',
            'examples': ['PG', 'CL', 'CLX', 'CHD', 'SPB', 'KMB', 'ENR'],
            'keywords': ['household', 'cleaning', 'consumer products', 'personal care'],
        },
        'personal_care': {
            'description': 'Personal care, cosmetics',
            'examples': ['EL', 'COTY', 'RBGLY', 'ELF', 'SKIN', 'ULTA', 'BRBR', 'REV'],
            'keywords': ['personal care', 'cosmetics', 'beauty', 'skincare', 'makeup'],
        },
        'tobacco': {
            'description': 'Tobacco, nicotine products',
            'examples': ['PM', 'MO', 'BTI', 'IMBBY', 'TPB', 'VGR'],
            'keywords': ['tobacco', 'cigarettes', 'nicotine', 'vaping', 'e-cigarette'],
        },
        'pet_products': {
            'description': 'Pet food, supplies',
            'examples': ['CHWY', 'FRPT', 'WOOF', 'TRUP', 'PET', 'IDXX', 'ZTS'],
            'keywords': ['pet', 'pet food', 'animal', 'veterinary', 'pet care'],
        },
    },

    # =========================================================================
    # MATERIALS & COMMODITIES
    # =========================================================================
    'materials': {
        'gold_miners': {
            'description': 'Gold mining companies',
            'examples': ['NEM', 'GOLD', 'AEM', 'FNV', 'WPM', 'KGC', 'GFI', 'AGI', 'AU', 'HL'],
            'keywords': ['gold', 'gold mining', 'precious metals', 'bullion'],
        },
        'silver_miners': {
            'description': 'Silver mining companies',
            'examples': ['AG', 'PAAS', 'HL', 'CDE', 'MAG', 'FSM', 'EXK', 'SILV'],
            'keywords': ['silver', 'silver mining', 'precious metals'],
        },
        'copper_miners': {
            'description': 'Copper mining companies',
            'examples': ['FCX', 'SCCO', 'TECK', 'IVPAF', 'HBM', 'ERO', 'COPX'],
            'keywords': ['copper', 'copper mining', 'base metals'],
        },
        'lithium_miners': {
            'description': 'Lithium mining, production',
            'examples': ['ALB', 'SQM', 'LAC', 'LTHM', 'PLL', 'SGML', 'ALTM', 'LTH'],
            'keywords': ['lithium', 'lithium mining', 'battery metals', 'brine'],
        },
        'rare_earth': {
            'description': 'Rare earth elements',
            'examples': ['MP', 'UUUU', 'TMRC', 'HREE', 'REMX'],
            'keywords': ['rare earth', 'REE', 'neodymium', 'permanent magnets'],
        },
        'diversified_mining': {
            'description': 'Diversified miners',
            'examples': ['BHP', 'RIO', 'VALE', 'GLNCY', 'AA', 'TECK'],
            'keywords': ['diversified mining', 'iron ore', 'mining', 'resources'],
        },
        'steel': {
            'description': 'Steel producers',
            'examples': ['NUE', 'X', 'CLF', 'STLD', 'CMC', 'RS', 'ATI', 'MT'],
            'keywords': ['steel', 'steel producer', 'iron', 'flat steel', 'mini mill'],
        },
        'aluminum': {
            'description': 'Aluminum producers',
            'examples': ['AA', 'CENX', 'ARNC', 'KALU', 'CSTM'],
            'keywords': ['aluminum', 'aluminium', 'bauxite', 'smelting'],
        },
        'chemicals_specialty': {
            'description': 'Specialty chemicals',
            'examples': ['APD', 'LIN', 'SHW', 'ECL', 'DD', 'PPG', 'IFF', 'ALB', 'ROL'],
            'keywords': ['specialty chemical', 'industrial gas', 'coatings', 'adhesives'],
        },
        'chemicals_commodity': {
            'description': 'Commodity chemicals',
            'examples': ['DOW', 'LYB', 'CE', 'WLK', 'OLN', 'HUN', 'TROX', 'KOP'],
            'keywords': ['commodity chemical', 'petrochemical', 'plastics', 'PVC'],
        },
        'fertilizers_agriculture': {
            'description': 'Fertilizers, ag chemicals',
            'examples': ['MOS', 'NTR', 'CF', 'FMC', 'CTVA', 'SMG', 'ICL', 'UAN'],
            'keywords': ['fertilizer', 'potash', 'nitrogen', 'agriculture', 'crop protection'],
        },
        'paper_packaging': {
            'description': 'Paper, packaging materials',
            'examples': ['IP', 'PKG', 'WRK', 'SON', 'BLL', 'CCK', 'SEE', 'BERY', 'GPK'],
            'keywords': ['paper', 'packaging', 'container', 'corrugated', 'flexible packaging'],
        },
    },

    # =========================================================================
    # UTILITIES
    # =========================================================================
    'utilities': {
        'electric_utilities': {
            'description': 'Regulated electric utilities',
            'examples': ['DUK', 'SO', 'D', 'AEP', 'XEL', 'EXC', 'PCG', 'ED', 'EIX', 'PPL'],
            'keywords': ['electric utility', 'power', 'regulated', 'IOU', 'transmission'],
        },
        'gas_utilities': {
            'description': 'Natural gas utilities',
            'examples': ['SRE', 'NI', 'ATO', 'OGS', 'NJR', 'SWX', 'SR', 'NFG'],
            'keywords': ['gas utility', 'natural gas distribution', 'LDC'],
        },
        'water_utilities': {
            'description': 'Water utilities',
            'examples': ['AWK', 'WTR', 'WTRG', 'CWT', 'SJW', 'MSEX', 'ARTNA'],
            'keywords': ['water utility', 'water treatment', 'wastewater'],
        },
        'multi_utilities': {
            'description': 'Multi-utility companies',
            'examples': ['DUK', 'NGG', 'AEE', 'WEC', 'CMS', 'DTE', 'ES', 'LNT'],
            'keywords': ['multi-utility', 'combination utility', 'integrated utility'],
        },
        'independent_power': {
            'description': 'Independent power producers',
            'examples': ['VST', 'NRG', 'CWEN', 'ORA', 'TAC', 'AES'],
            'keywords': ['IPP', 'independent power', 'merchant power', 'renewable'],
        },
    },

    # =========================================================================
    # COMMUNICATION SERVICES
    # =========================================================================
    'communication_services': {
        'telecom_wireless': {
            'description': 'Wireless carriers',
            'examples': ['T', 'VZ', 'TMUS', 'USM', 'SHEN'],
            'keywords': ['wireless', 'mobile', 'cellular', 'telecom', '5G'],
        },
        'telecom_cable': {
            'description': 'Cable, broadband providers',
            'examples': ['CMCSA', 'CHTR', 'LBRDA', 'CABO', 'ATUS', 'FYBR'],
            'keywords': ['cable', 'broadband', 'internet', 'fiber', 'MSO'],
        },
        'media_broadcast': {
            'description': 'Broadcast media, TV networks',
            'examples': ['FOX', 'PARA', 'WBD', 'NXST', 'GTN', 'SBGI', 'TGNA'],
            'keywords': ['broadcast', 'TV network', 'media', 'local TV'],
        },
        'publishing': {
            'description': 'Publishing, newspapers',
            'examples': ['NYT', 'NWSA', 'GCI', 'LEE', 'DJCO'],
            'keywords': ['publishing', 'newspaper', 'media', 'digital news'],
        },
        'advertising_agencies': {
            'description': 'Ad agencies, holding companies',
            'examples': ['OMC', 'IPG', 'WPP', 'PUBGY'],
            'keywords': ['advertising', 'agency', 'media buying', 'creative'],
        },
    },

    # =========================================================================
    # EMERGING THEMES / SPECIAL SITUATIONS
    # =========================================================================
    'emerging_themes': {
        'meme_stocks': {
            'description': 'High social media attention stocks',
            'examples': ['GME', 'AMC', 'BB', 'KOSS', 'BBBY', 'EXPR', 'NAKD'],
            'keywords': ['meme', 'Reddit', 'WSB', 'retail', 'short squeeze'],
        },
        'spacs': {
            'description': 'SPACs and de-SPAC companies',
            'examples': [],  # Dynamic - varies by market
            'keywords': ['SPAC', 'blank check', 'de-SPAC', 'merger'],
        },
        'china_adr': {
            'description': 'Chinese ADRs',
            'examples': ['BABA', 'JD', 'PDD', 'BIDU', 'NIO', 'LI', 'XPEV', 'BILI', 'TME', 'IQ'],
            'keywords': ['China', 'ADR', 'Chinese tech', 'VIE', 'delisting'],
        },
        'reopening_plays': {
            'description': 'Post-COVID recovery plays',
            'examples': ['DAL', 'UAL', 'CCL', 'RCL', 'MAR', 'LYV', 'BKNG'],
            'keywords': ['reopening', 'recovery', 'travel', 'leisure'],
        },
        'short_squeeze': {
            'description': 'High short interest candidates',
            'examples': [],  # Dynamic - based on short interest data
            'keywords': ['short squeeze', 'short interest', 'borrow', 'days to cover'],
        },
        'insider_buying': {
            'description': 'Significant insider purchases',
            'examples': [],  # Dynamic - based on Form 4 filings
            'keywords': ['insider buying', 'Form 4', 'CEO purchase', 'director'],
        },
    },
}


# =============================================================================
# TICKER TO INDUSTRY LOOKUP (Reverse mapping for fast lookups)
# =============================================================================

def build_ticker_to_industry_map() -> Dict[str, Dict]:
    """Build a reverse mapping from ticker to industry information."""
    ticker_map = {}
    for sector, industries in INDUSTRY_CLASSIFICATION.items():
        for industry, details in industries.items():
            for ticker in details.get('examples', []):
                if ticker not in ticker_map:
                    ticker_map[ticker] = {
                        'sector': sector,
                        'industry': industry,
                        'description': details['description'],
                    }
    return ticker_map


TICKER_TO_INDUSTRY = build_ticker_to_industry_map()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_industry_for_ticker(ticker: str) -> Optional[Dict]:
    """
    Get industry classification for a ticker.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Dictionary with sector, industry, description or None if not found
    """
    return TICKER_TO_INDUSTRY.get(ticker.upper())


def get_peers_for_ticker(ticker: str, max_peers: int = 20) -> List[str]:
    """
    Get peer stocks in the same industry.

    Args:
        ticker: Stock ticker symbol
        max_peers: Maximum number of peers to return

    Returns:
        List of peer ticker symbols
    """
    info = get_industry_for_ticker(ticker)
    if not info:
        return []

    sector = info['sector']
    industry = info['industry']

    examples = INDUSTRY_CLASSIFICATION.get(sector, {}).get(industry, {}).get('examples', [])

    # Return peers excluding the input ticker
    peers = [t for t in examples if t.upper() != ticker.upper()]
    return peers[:max_peers]


def get_all_industries() -> List[str]:
    """Get list of all industry names."""
    industries = []
    for sector, sector_industries in INDUSTRY_CLASSIFICATION.items():
        for industry in sector_industries.keys():
            industries.append(f"{sector}/{industry}")
    return sorted(industries)


def get_all_sectors() -> List[str]:
    """Get list of all sector names."""
    return list(INDUSTRY_CLASSIFICATION.keys())


def search_by_keyword(keyword: str) -> List[Dict]:
    """
    Search for industries matching a keyword.

    Args:
        keyword: Search term

    Returns:
        List of matching industries with their details
    """
    keyword_lower = keyword.lower()
    results = []

    for sector, industries in INDUSTRY_CLASSIFICATION.items():
        for industry, details in industries.items():
            keywords = details.get('keywords', [])
            if any(keyword_lower in kw.lower() for kw in keywords):
                results.append({
                    'sector': sector,
                    'industry': industry,
                    'description': details['description'],
                    'examples': details['examples'],
                })

    return results


def get_industry_tickers(sector: str, industry: str) -> List[str]:
    """
    Get all tickers for a specific industry.

    Args:
        sector: Sector name
        industry: Industry name

    Returns:
        List of ticker symbols
    """
    return INDUSTRY_CLASSIFICATION.get(sector, {}).get(industry, {}).get('examples', [])


# =============================================================================
# INDUSTRY STATISTICS
# =============================================================================

def get_industry_stats() -> Dict:
    """Get statistics about the industry classification."""
    total_industries = 0
    total_tickers = 0
    tickers_per_sector = {}

    for sector, industries in INDUSTRY_CLASSIFICATION.items():
        sector_tickers = 0
        for industry, details in industries.items():
            total_industries += 1
            ticker_count = len(details.get('examples', []))
            total_tickers += ticker_count
            sector_tickers += ticker_count
        tickers_per_sector[sector] = sector_tickers

    return {
        'total_sectors': len(INDUSTRY_CLASSIFICATION),
        'total_industries': total_industries,
        'total_tickers': total_tickers,
        'unique_tickers': len(TICKER_TO_INDUSTRY),
        'tickers_per_sector': tickers_per_sector,
    }


if __name__ == '__main__':
    # Print industry statistics
    stats = get_industry_stats()
    print("Industry Classification Statistics:")
    print(f"  Total Sectors: {stats['total_sectors']}")
    print(f"  Total Industries: {stats['total_industries']}")
    print(f"  Unique Tickers: {stats['unique_tickers']}")
    print("\nTickers per Sector:")
    for sector, count in sorted(stats['tickers_per_sector'].items(), key=lambda x: -x[1]):
        print(f"  {sector}: {count}")
