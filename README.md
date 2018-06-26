# Applied Project - Loan Granting
by Wenbo Ma

## Introduction

### Overview

Machine learning plays an important role in choosing if granting a loan. In this project, we have access to a bank's loan application data. Our goal is to leverage machine learning model to come up with a loan strategy that would be better than the bank's current one.

### Data

The data contains all loans asked to the bank, whether the bank decided to grant it and finally, whether the borrower managed to repay it. We also have info about the borrowers at the time they ask for a loan.

### Rule

To compare the profitability of our model to that of the bank's current strategy, the following rule is assumed:

  * If our model grant a loan and it doesn't get repaid, we lose 1
  * If our model grant a loan and it gets repaid, we gain 1
  * If our model dosen't grant the loan, we gain 0

## Model Building Procedure and Analysis

Please see the project report and code link for details.

[Project Report](https://github.com/wenbo5565/AppliedProject_GrantingLoan/blob/master/Project_Report__Loan_Granting.pdf)

[Project Code](https://github.com/wenbo5565/AppliedProject_GrantingLoan/blob/master/GrantingLoan%20core.py)

## Result and Proposed Strategy

### Profitability

Based on the scoring assumption from **Rule** above, our model dominates the bank's current strategy (calculated based on the data). The performance are as follows:
