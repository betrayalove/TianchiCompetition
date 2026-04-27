---
title: "算法大赛-天池大赛-阿里云的赛制"
source: "https://tianchi.aliyun.com/competition/entrance/231574/information"
author:
published:
created: 2026-04-27
description: "天池大数据竞赛,是由阿里巴巴集团主办,面向全球科研工作者的高端算法竞赛。通过开放海量数据和分布式计算资源,大赛让所有参与者有机会运用其设计的算法解决各类社会问题或业务问题。欢迎来大家来天池参与天池大数据竞赛,进行真实业务场景演练,参与天池大赛还有机会获得百万奖金池。"
tags:
  - "clippings"
---
### 数据下载

1\. Season 1.  
**1.1 Training data (weibo\_train\_data).  
**We sample on users and take out the original weibos of each target user in half a year (from 20140701 to 20141231). User id and Weibo id are encrypted.

| Attribute | Description |
| --- | --- |
| uid | user id. Sampled and encrypted |
| mid | weibo id. Sampled and encrypted |
| time | post time. Format YYYYMMDD |
| forward\_count | amount of forward within one week after posting |
| comment\_count | amount of comment within one week after posting |
| like\_count | amount of comment within one week after posting |
| content | weibo content |

**1.2 Predicting data (weibo\_predict\_data).  
**Predicting data is from 20150101 to 20150131.

| Attribute | Description |
| --- | --- |
| uid | user id. Sampled and encrypted |
| mid | weibo id. Sampled and encrypted |
| time | post time. Format YYYYMMDD |
| content | weibo content |

**1.3 Predicting result (example: weibo\_result\_data)**  
Participants are required to predict the cumulated forwarding, commenting and liking amount within one week after the posting of each weibo in weibo\_predict\_data and submit this result. Submitting format is as following:

| Attribute | Description |
| --- | --- |
| uid | user id. Sampled and encrypted |
| mid | weibo id. Sampled and encrypted |
| forward\_count | amount of forward within one week after posting |
| comment\_count | amount of comment within one week after posting |
| like\_count | amount of comment within one week after posting |

**3\. Evaluation  
**![](https://gtms04.alicdn.com/tps/i4/TB15YDlIVXXXXXPXFXXaNlTRpXX-617-236.jpg) 00.

Participants should submit the prediction results into a txt file (file name must be within 20 characters) with specified format:  
uid mid forward\_count, comment\_count, like\_count.  
For example:

![](https://gtms01.alicdn.com/tps/i1/TB19YkSIVXXXXbjaXXX2m8D4VXX-726-97.png)

**Contents Page** 空

<iframe src="https://free.aliyun.com/smarter-engine?at_iframe=1"></iframe><iframe src="about:blank"></iframe>