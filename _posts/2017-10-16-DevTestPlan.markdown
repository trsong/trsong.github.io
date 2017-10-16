---
layout: post
title:  "Dev Test Plan"
date:   2017-10-16 22:36:32 -0700
categories: Test
---
* This will become a table of contents (this text will be scraped).
{:toc}

### What to include in a Dev Test Plan

| Item                                                              | Explanation | 
|:----------------------------------------------------------------- |:--------------- |
| Description                                                       | What is the feature intended for? How will it contribute to the overall business value? | 
| Happy Paths                                                       | The top user workflows of the feature that demonstrates the business values |
| Related Exisiting Feature Test Cases                              | E.g. If we create a new chart type, we should test Export PDF for this new chart type |
| Changes to expected behaviour for exisiting features              | E.g. Key groups as a new feature which changed some expected behaviour of Share Plans / Analyses |
| Known limitations                                                 | E.g. Can only do one Group-By. Or can only add primary metric as a filter |
| Known dependencies / risks                                        | How will the new code / code change get surfaced through the UI? Basically, full understanding of the code path and its riskes. |
| Migration required / New BP required / Impact on exisiting content| Self-explaining. Good Example are Overlay and TCOW |

