# Applied Project - Loan Granting
by Wenbo Ma

## Introduction

### Overview

Loan interest charges is an important revenue stream for commerial banks and other financial institutions. Loan application prediction is a key component in the loan management process. In this project, we develop a machine learnning model to predict whether a loan application will be fully paid back in the further based a bank's historical data. We show that our model's performance is better than the bank's current strategy. Additionally, we propose a new strategy to further increase revenues based on our analysis. 

### Data

The data contains all loan applications submitted to a bank, whether the bank decided to grant it and finally, whether the borrower managed to repay it. We also have info about the borrowers at the time they ask for a loan.

### Rule

To compare the profitability of our model to that of the bank's current strategy, the following rule is assumed:

  * If our model grant a loan and it doesn't get repaid, we lose 1
  * If our model grant a loan and it gets repaid, we gain 1
  * If our model dosen't grant the loan, we gain 0

## Model Building Procedure and Analysis

Please see the project report and code link for details.

[Project Report](https://github.com/wenbo5565/AppliedProject_GrantingLoan/blob/master/Project_Report__Loan_Granting.pdf) --- Data Description, Exploratory Analysis and Model Building Procedure

[Project Code](https://github.com/wenbo5565/AppliedProject_GrantingLoan/blob/master/GrantingLoan%20core.py) --- All Python Code for the Project

## Result and Proposed Strategy

### Profitability

Based on the scoring assumption from **Rule** above, our model dominates the bank's current strategy (calculated based on the data). The performance is shown below.

| Model | Total Score | Mean Score per Loan
| --- | --- | ---- |
| Bank's Current Model | 1376 | 0.289 |
| Our Model | 2699 | 0.566 |

### Strategy


