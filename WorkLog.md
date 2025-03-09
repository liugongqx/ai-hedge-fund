

# Goal
1. Fact check, identify data source ground truth
2. Algorithem/trading understanding
3. Backtest code learning
4. Local llm onboarding (Couple git branch already have it)

# Log
3/9
1. Ask ground truth MSFT to discord
2. Change period from ttm to quarterly but no change on result.Change to "annual" fix the issue, ttm is "railing Twelve Months", this is more up to date data but it may include some weak months.
3. Change backtest period from **1** day to **7** days and only execute on **Monday**, This means the sample time will be every week. This may not the right way as it will reduce the graunulatiy and may introduce error for an algorithem, however, this make evaludate faster
4. Integrate DeepSeekAPI and this is prefered as it is **much cheaper**! 

3/8
1. Understand code structure
2. Onboard Open AI API
3. Investigate MSFT Ground truth low issue. As the follow data is much lower than public number
"Revenue Growth: 2.99%, Earnings Growth: 2.47%"