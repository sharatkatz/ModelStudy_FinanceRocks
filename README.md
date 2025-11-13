# ModelTrain_FR

![PyPI version](https://img.shields.io/pypi/v/ModelStudy_FR.svg)
[![Documentation Status](https://readthedocs.org/projects/ModelStudy_FR/badge/?version=latest)](https://ModelStudy_FR.readthedocs.io/en/latest/?version=latest)

Codes required to preprocess the input data and execute a modeling pipeline on the top of it.

* PyPI package: https://pypi.org/project/ModelStudy_FR/
* Free software: MIT License
* Documentation: https://ModelStudy_FR.readthedocs.io.

## Features

* TODO


## Flow:
1. Import necessary modules and classes.
2. Define global variables for file paths.
3. Set up a temporary directory for storing plots.
4. The module is designed to be imported and used by other scripts or run as a standalone
    script.
5. When run as a standalone script, it initializes the plot directory.
6. Input files are expected to be located in a specific directory structure.
7. We model Total Revenue (Sum of all revenue components): $Y = \text{Total Discounted Revenue} = \sum (\text{All } line\_total\_discounted\_vat\_0\_rev\_\text{...})$$
as the outcome variable.
8. The package variable is "package".
9. Predictor variables include various revenue components and customer attributes.
10. All the meta information is classified under four types of variables:
    - Customer/Company Profile: 
        ["id", 
        "package", 
        "accounting_office_id", 
        "company_type_label", 
        "tol_1_eng", 
        "tol_2_eng", 
        "headcount_class", 
        "revenue_class", 
        "ao_revenue_class", 
        "ao_headcount_class"]

    - Usage/Activity Metrics (over 12 months): 
        ["total_records_months_used", 
         "total_records_mean", 
         "total_records_sum", 
         "total_SI_PI_vouchers_months_used", 
         "total_SI_PI_vouchers_mean", 
         "total_SI_PI_vouchers_sum", 
         "record_count_salary_months_used", 
         "record_count_salary_mean"]

    - Add-on/Feature Usage (Binary/Mobile): 
        ["add_api", 
         "add_bank_account", 
         "add_contract_invoicing", 
         "add_cust_invoice", 
         "add_ext_dimensions", 
         "add_inventory", 
         "add_junior", 
         "add_mobile", 
         "add_sftp", 
         "mobile_user_count"]

    - Revenue Metrics (Before and After Discounts): 
        ["line_total_vat_0_rev_package", 
         "line_total_vat_0_rev_ex_vouchers", 
         "line_total_vat_0_rev_ex_employees", 
         "line_total_vat_0_rev_integrations", 
         "line_total_vat_0_rev_mobile", 
         "line_total_vat_0_rev_addon", 
         "line_total_vat_0_rev_trx", 
         "line_total_discounted_vat_0_rev_package", 
         "line_total_discounted_vat_0_rev_ex_vouchers", 
         "line_total_discounted_vat_0_rev_ex_employees", 
         "line_total_discounted_vat_0_rev_integrations", 
         "line_total_discounted_vat_0_rev_mobile", 
         "line_total_discounted_vat_0_rev_addon", 
         "line_total_discounted_vat_0_rev_trx"]

 11. All the columns and their data descriptors is provided as:

| COLUMNS                                      | DESCRIPTION                                                                                                                    |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| id                                           | unique end-customer id                                                                                                         |
| product_package                              | product package                                                                                                                |
| accounting_office_id                         | the id of the customer’s accounting office                                                                                     |
| company_type_label                           | the legal type of the customer’s business (e.g. limited liability (LTD), sole proprietor (TMI), etc.)                          |
| tol_1_eng                                    | industry category of the customer’s business (level 1)                                                                         |
| tol_2_eng                                    | industry category of the customer’s business (level 2)                                                                         |
| headcount_class                              | the revenue and customer company/install falls into based on the number of employees it has                                    |
| revenue_class                                | the revenue bracket the customer company falls into based on its annual revenue                                                |
| record_count_accounting_office               | the number of clients of the customer’s accounting office                                                                      |
| total_records_months_used                    | number of months the customer has used a voucher during the 12-month period                                                    |
| total_records_mean                           | average number of vouchers used per month in 12 months                                                                         |
| total_records_sum                            | total number of vouchers used in 12 months                                                                                     |
| total_SI_PI_vouchers_months_used             | number of months the customer generated sales and purchase invoices during the 12-month period                                 |
| total_SI_PI_vouchers_mean                    | average number of sales and purchase invoices used per month in 12 months                                                      |
| total_SI_PI_vouchers_sum                     | total number of sales and purchase invoices used in 12 months                                                                  |
| record_count_salary_months_used              | number of months the customer has used payroll functionality during the past 12 months                                         |
| record_count_salary_mean                     | average number of employees on payroll during past 12 months                                                                   |
| add_api                                      | 1. Indicates if the customer is using one of the add-on features such as api, credit search, bank account, consolidation, etc. |
|                                              | 2. "add_api" and "add_sftp" denote integration usage                                       |                                   |
|                                              | 3. "add_mobile" denotes the usage of the mobile app                                                                            |
| add_stfp                                     |                                                                                                                                |
| add_bank_account                             |                                                                                                                                |
| add_contract_invoicing                       |                                                                                                                                |
| add_cust_invoice                             |                                                                                                                                |
| add_ext_dimensions                           |                                                                                                                                |
| add_inventory                                |                                                                                                                                |
| add_junior                                   |                                                                                                                                |
| add_mobile                                   |                                                                                                                                |
| mobile_user_count                            | number of users using the mobile app if the customer has enabled the mobile app                                                |
| line_total_vat_0_rev_package                 | revenue from package fees (before discounts)                                                                                   |
| line_total_vat_0_rev_vouchers                | revenue from exceeded voucher fees (before discounts)                                                                          |
| line_total_vat_0_rev_employees               | revenue from exceeded payroll fees (before discounts)                                                                          |
| line_total_vat_0_rev_integrations            | revenue from integrations (before discounts)                                                                                   |
| line_total_vat_0_rev_mobile                  | revenue from mobile (before discounts)                                                                                         |
| line_total_vat_0_rev_addon                   | revenue from other add-ons (before discounts)                                                                                  |
| line_total_discounted_vat_0_rev_package      | revenue from package fees (after discounts)                                                                                    |
| line_total_discounted_vat_0_rev_vouchers     | revenue from exceeded voucher fees (after discounts)                                                                           |
| line_total_discounted_vat_0_rev_employees    | revenue from exceeded payroll fees (after discounts)                                                                           |
| line_total_discounted_vat_0_rev_integrations | revenue from integrations (after discounts)                                                                                    |
| line_total_discounted_vat_0_rev_mobile       | revenue from mobile (after discounts)                                                                                          |
| line_total_discounted_vat_0_rev_addon        | revenue from other add-ons (after discounts)                                                                                   | 



---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 🛠️ Recommended Independent Variables for Total Revenue Model
The goal is to use variables that describe who the customer is (profile), how big they are (size), and what they use (adoption/usage), but not the direct outcome of the revenue calculation.

## 1. Customer Profile & Segmentation (Categorical)
These variables segment your customer base and should be entered into the model as dummy (one-hot encoded) variables.

package: (Crucial, as analyzed previously) Captures the price tier premium.

company_type_label: Distinguishes legal structure (e.g., LTD, TMI), which may influence spending/compliance needs.

tol_1_eng: Captures industry-specific revenue potential or needs.

headcount_class / revenue_class: (Use one, or both if they are not perfectly correlated) Captures the inherent size and scale of the business.

## 2. Usage & Engagement (Continuous)
These variables capture the customer's operational volume and activity, which drives variable revenue (overage).

total_records_sum or total_records_mean: Measures overall bookkeeping activity, which should correlate strongly with total value.

total_SI_PI_vouchers_sum: Measures sales and purchase invoice volume—a key indicator of business activity and billing potential.

total_records_months_used / total_SI_PI_vouchers_months_used: Measures customer consistency and engagement over the year.

record_count_salary_mean: Measures the average payroll load, predicting payroll-related revenue.

## 3. Feature Adoption (Binary/Count)
These indicate the customer's willingness to use high-value add-ons.

All add_ binary variables (e.g., add_api, add_bank_account, add_mobile, etc.): These should strongly predict integration/addon revenue.

mobile_user_count: A granular measure of mobile engagement for those who've adopted it.

## 4. Accounting Office (AO) Influence
These capture the impact of the channel/partner.

ao_headcount_class / ao_revenue_class: Captures the size and presumably the sophistication of the customer's accounting partner.

## 5. tol_2_.... column names are renamed as per the following scheme:
| Original Name | Renamed To |
| :--- | :--- |
| tol_2_eng_activities_auxiliary_to_financial_services_and_insurance_activities | tol_2_RENAMED_1 |
| tol_2_eng_activities_of_head_offices__management_consultancy_activities | tol_2_RENAMED_2 |
| tol_2_eng_activities_of_membership_organisations | tol_2_RENAMED_3 |
| tol_2_eng_advertising_and_market_research | tol_2_RENAMED_4 |
| tol_2_eng_air_transport | tol_2_RENAMED_5 |
| tol_2_eng_architectural_and_engineering_activities__technical_testing_and_analysis | tol_2_RENAMED_6 |
| tol_2_eng_civil_engineering | tol_2_RENAMED_7 |
| tol_2_eng_computer_programming__consultancy_and_related_activities | tol_2_RENAMED_8 |
| tol_2_eng_construction_of_buildings | tol_2_RENAMED_9 |
| tol_2_eng_creative__arts_and_entertainment_activities | tol_2_RENAMED_10 |
| tol_2_eng_crop_and_animal_production__hunting_and_related_service_activities | tol_2_RENAMED_11 |
| tol_2_eng_education | tol_2_RENAMED_12 |
| tol_2_eng_electricity__gas__steam_and_air_conditioning_supply | tol_2_RENAMED_13 |
| tol_2_eng_employment_activities | tol_2_RENAMED_14 |
| tol_2_eng_financial_service_activities__except_insurance_and_pension_funding | tol_2_RENAMED_15 |
| tol_2_eng_fishing_and_aquaculture | tol_2_RENAMED_16 |
| tol_2_eng_food_and_beverage_service_activities | tol_2_RENAMED_17 |
| tol_2_eng_forestry_and_logging | tol_2_RENAMED_18 |
| tol_2_eng_human_health_activities | tol_2_RENAMED_19 |
| tol_2_eng_industry_unknown | tol_2_RENAMED_20 |
| tol_2_eng_information_service_activities | tol_2_RENAMED_21 |
| tol_2_eng_insurance__reinsurance_and_pension_funding__except_compulsory_social_security | tol_2_RENAMED_22 |
| tol_2_eng_land_transport_and_transport_via_pipelines | tol_2_RENAMED_23 |
| tol_2_eng_legal_and_accounting_activities | tol_2_RENAMED_24 |
| tol_2_eng_libraries__archives__museums_and_other_cultural_activities | tol_2_RENAMED_25 |
| tol_2_eng_manufacture_of_beverages | tol_2_RENAMED_26 |
| tol_2_eng_manufacture_of_chemicals_and_chemical_products | tol_2_RENAMED_27 |
| tol_2_eng_manufacture_of_computer__electronic_and_optical_products | tol_2_RENAMED_28 |
| tol_2_eng_manufacture_of_electrical_equipment | tol_2_RENAMED_29 |
| tol_2_eng_manufacture_of_fabricated_metal_products__except_machinery_and_equipment | tol_2_RENAMED_30 |
| tol_2_eng_manufacture_of_food_products | tol_2_RENAMED_31 |
| tol_2_eng_manufacture_of_furniture | tol_2_RENAMED_32 |
| tol_2_eng_manufacture_of_leather_and_related_products | tol_2_RENAMED_33 |
| tol_2_eng_manufacture_of_machinery_and_equipment_n.e.c. | tol_2_RENAMED_34 |
| tol_2_eng_manufacture_of_motor_vehicles__trailers_and_semi_trailers | tol_2_RENAMED_35 |
| tol_2_eng_manufacture_of_other_non_metallic_mineral_products | tol_2_RENAMED_36 |
| tol_2_eng_manufacture_of_other_transport_equipment | tol_2_RENAMED_37 |
| tol_2_eng_manufacture_of_paper_and_paper_products | tol_2_RENAMED_38 |
| tol_2_eng_manufacture_of_rubber_and_plastic_products | tol_2_RENAMED_39 |
| tol_2_eng_manufacture_of_textiles | tol_2_RENAMED_40 |
| tol_2_eng_manufacture_of_wearing_apparel | tol_2_RENAMED_41 |
| tol_2_eng_manufacture_of_wood_and_of_products_of_wood_and_cork__except_furniture__manufacture_of_articles_of_straw_and_plaiting_materials | tol_2_RENAMED_42 |
| tol_2_eng_mining_support_service_activities | tol_2_RENAMED_43 |
| tol_2_eng_motion_picture__video_and_television_programme_production__sound_recording_and_music_publishing_activities | tol_2_RENAMED_44 |
| tol_2_eng_office_administrative__office_support_and_other_business_support_activities | tol_2_RENAMED_45 |
| tol_2_eng_other_manufacturing | tol_2_RENAMED_46 |
| tol_2_eng_other_mining_and_quarrying | tol_2_RENAMED_47 |
| tol_2_eng_other_personal_service_activities | tol_2_RENAMED_48 |
| tol_2_eng_other_professional__scientific_and_technical_activities | tol_2_RENAMED_49 |
| tol_2_eng_postal_and_courier_activities | tol_2_RENAMED_50 |
| tol_2_eng_printing_and_reproduction_of_recorded_media | tol_2_RENAMED_51 |
| tol_2_eng_public_admin_social_insurance | tol_2_RENAMED_52 |
| tol_2_eng_publishing_activities | tol_2_RENAMED_53 |
| tol_2_eng_real_estate_activities | tol_2_RENAMED_54 |
| tol_2_eng_rental_and_leasing_activities | tol_2_RENAMED_55 |
| tol_2_eng_repair_and_installation_of_machinery_and_equipment | tol_2_RENAMED_56 |
| tol_2_eng_repair_of_computers_and_personal_and_household_goods | tol_2_RENAMED_57 |
| tol_2_eng_residential_care_activities | tol_2_RENAMED_58 |
| tol_2_eng_retail_trade__except_of_motor_vehicles_and_motorcycles | tol_2_RENAMED_59 |
| tol_2_eng_scientific_research_and_development | tol_2_RENAMED_60 |
| tol_2_eng_security_and_investigation_activities | tol_2_RENAMED_61 |
| tol_2_eng_services_to_buildings_and_landscape_activities | tol_2_RENAMED_62 |
| tol_2_eng_sewerage | tol_2_RENAMED_63 |
| tol_2_eng_social_work_activities_without_accommodation | tol_2_RENAMED_64 |
| tol_2_eng_specialised_construction_activities | tol_2_RENAMED_65 |
| tol_2_eng_sports_activities_and_amusement_and_recreation_activities | tol_2_RENAMED_66 |
| tol_2_eng_telecommunications | tol_2_RENAMED_67 |
| tol_2_eng_travel_agency__tour_operator_and_other_reservation_service_and_related_activities | tol_2_RENAMED_68 |
| tol_2_eng_veterinary_activities | tol_2_RENAMED_69 |
| tol_2_eng_warehousing_and_support_activities_for_transportation | tol_2_RENAMED_70 |
| tol_2_eng_waste_collection__treatment_and_disposal_activities__materials_recovery | tol_2_RENAMED_71 |
| tol_2_eng_water_collection__treatment_and_supply | tol_2_RENAMED_72 |
| tol_2_eng_water_transport | tol_2_RENAMED_73 |
| tol_2_eng_wholesale_and_retail_trade_and_repair_of_motor_vehicles_and_motorcycles | tol_2_RENAMED_74 |
| tol_2_eng_wholesale_trade__except_of_motor_vehicles_and_motorcycles | tol_2_RENAMED_75 | 

1. Preferred Model: Log-Linear/Log-Log Form (Elasticity-Focused)This form is better for understanding the percentage impact of a change in usage (the independent variables) on Total Revenue (the dependent variable).
Model Form:$$\ln(Y) = \beta_0 + \sum_{j=1}^{K} \beta_j X_j + \sum_{k=1}^{P-1} \gamma_k D_k + \epsilon$$
Interpretation of Coefficients:
    - For $X_j$ (Continuous/Level): A one-unit increase in $X_j$ leads to a $(\beta_j \times 100)\%$ change in $Y$ (Total Revenue).
    - For $\ln(X_j)$ (Log-Log): A 1% increase in $X_j$ (e.g., vouchers used) leads to a $\beta_j\%$ change in $Y$ (Total Revenue). This $\beta_j$ is the elasticity.
    - For $D_k$ (Dummy): Being in category $D_k$ (e.g., $\text{package\_7}$) leads to a $(e^{\gamma_k} - 1) \times 100\%$ change in $Y$ compared to the baseline category.

2. Standard Model: Multiple Linear Regression (Absolute Change Focused)This simpler form predicts the absolute Euro change in revenue.
Model Form:$$Y = \beta_0 + \sum_{j=1}^{K} \beta_j X_j + \sum_{k=1}^{P-1} \gamma_k D_k + \epsilon$$
Variable Definitions:
 - $Y$ is now the Total Revenue (in its raw, non-log-transformed Euro value). The independent variables $X_j$ and $D_k$ remain the same.Interpretation of Coefficients:
 - For $X_j$ (Continuous): A one-unit increase in $X_j$ leads to a $\beta_j$ (in €) change in $Y$ (Total Revenue).For $D_k$ (Dummy): Being in category $D_k$ leads to a $\gamma_k$ (in €) difference in $Y$ compared to the baseline category.
 
Recommendation: Start with the Log-Linear/Log-Log Model (Form 1) as it naturally handles the non-linear relationship often seen between usage/size and revenue, and directly provides the elasticity measures you are interested in.

---------------------------------------------------------------------------------------
Based on https://www.statsmodels.org/stable/glm.html:
The most appropriate distribution family to consider for modeling your Total Revenue (if you were to use a Generalized Linear Model (GLM) instead of the Log-Log/OLS approach) is the Gamma distribution or, more comprehensively, the Tweedie distribution.
Here is the reasoning based on the characteristics of your revenue data:
🔬 Recommended Distribution Family: Gamma
Why the Gamma Distribution?The Gamma distribution (available in statsmodels.genmod.families.family.Gamma) is the standard choice for modeling non-negative, continuous, and right-skewed variables, which perfectly describes financial data like customer revenue.
- Non-Negativity: Revenue ($Y$) cannot be negative. The Gamma distribution is defined only for $Y > 0$.
- Skewness: Revenue is typically heavily right-skewed (many customers with low revenue, a few outliers with extremely high revenue). The Gamma distribution is asymmetric and designed to handle this high skewness.
- Variance Relationship: The Gamma distribution implies that the variance of the residuals increases as the mean predicted revenue increases ($\text{Var}[Y] \propto \mu^2$). This is a common characteristic of revenue data: high-value customers tend to have much larger and more volatile errors than low-value customers.The Link Function

When using the Gamma family, the typical link function used is the log link (link=sm.families.links.log()). This maintains the connection to your earlier Log-Linear approach:$$\ln(\mu_i) = \beta_0 + \sum \beta_j X_j$$Here, $\mu_i$ is the mean expected revenue for customer $i$. Using the log link ensures that the predicted revenue ($\mu_i$) will always be positive, which is consistent with the nature of revenue.
