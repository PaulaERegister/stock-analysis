"""
A set of functions for statistical stock market graphing and analysis using time series analysis - to
calculate moving averages, trading strategies using moving averages, formulation of exit strategies,
and strategy evaluation using backtesting - primarily using quandl, pandas, numpy, and matplotlib.
Examples generalized from CS5160 Introduction to Data Science at University of Utah.
Source: https://ntguardian.wordpress.com/2018/07/17/stock-data-analysis-python-v2/
"""

import quandl
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, WeekdayLocator, DayLocator, MONDAY
import matplotlib.dates as dates
from mpl_finance import candlestick_ohlc

api_key = "itSiso5s5gexytWVr4Aw"

def moving_average_crossover_strategy(days, stk, start_date, end_date):
    """
    :param days: An array of the day ranges to calculate a moving average from
    :param stk: The particular stock to examine
    :param start_date: The starting date
    :param end_date: The end date
    :return: The stock
    """
    ds = [d + "d" for d in days]
    start_range = ds[0]
    end_range = ds[1]
    range = (start_range+"-"+end_range)
    stk[range] = stk[start_range] - stk[end_range]
    print(stk.tail())
    stk["Regime"] = np.where(stk[range] > 0,1,0)
    # We have 1's for bullish regimes and 0's for everything else
    #replace bearish regimes's values with -1, and to maintain the rest of the vector,
    # the second argument is stk["Regime"]
    stk["Regime"] = np.where(stk[range] < 0,-1, stk["Regime"])
    #stk.loc[start_date:end_date,"Regime"].plot(ylim=(-2,2)).axhline(y=0, color="black",lw=2)
    #stk["Regime"].plot(ylim=(-2, 2)).axhline(y=0, color="black", lw=2)
    count = stk["Regime"].value_counts().tolist()
    print("The market was bearish for " + str(count[0]) + " days.")
    print("The market was bullish for " + str(count[1]) + " days.")
    print("The market was neutral for " + str(count[2]) + " days.")
    get_signals(stk)
def get_signals(stk):
    """
    s_t = sign(r_t - r_(t-1)) where s_t is an element of {-1,0,1} where
    -1 indicates sell
    1 indicates buy
    0 indicates no action
    :param stk:
    :return:
    """
    # To ensure that all trades close out, temporarily change the regime of the last row to 0
    regime_orig = stk.loc[:,"Regime"].iloc[-1]
    stk.loc[:,"Regime"].iloc[-1] = 0
    stk["Signal"] = np.sign(stk["Regime"] - stk["Regime"].shift(1))
    # restore data
    stk.loc[:,"Regime"].iloc[-1] = regime_orig
    print(stk.tail())
    #stk["Signal"].plot(ylim=(-2,2))
    count = stk["Signal"].value_counts().tolist()
    print("We would buy stock " + str(count[2]) + " times.")
    print("We would sell stock " + str(count[1]) + " times.")
    print("We do nothing " + str(count[0]) + " times.")
    identify_prices_at_close(stk)
def identify_prices_at_close(stk):
    print(stk.loc[stk["Signal"] == 1, "Close"])
    print(stk.loc[stk["Signal"] == -1, "Close"])
    stk_signals = pd.concat([
        pd.DataFrame({"Price": stk.loc[stk["Signal"] == 1, "Adj. Close"],
                      "Regime": stk.loc[stk["Signal"] == 1, "Regime"],
                      "Signal": "Buy"}),
        pd.DataFrame({"Price": stk.loc[stk["Signal"] == -1, "Adj. Close"],
                      "Regime": stk.loc[stk["Signal"] == -1, "Regime"],
                      "Signal": "Sell"}),
    ])
    stk_signals.sort_index(inplace = True)
    print(stk_signals)
    long_trade_profitability(stk_signals, stk)
def long_trade_profitability(stk_signals, stk):
    stk_long_profits = pd.DataFrame({
        "Price": stk_signals.loc[(stk_signals["Signal"] == "Buy") &
                                 stk_signals["Regime"] == 1, "Price"],
        "Profit": pd.Series(stk_signals["Price"] - stk_signals["Price"].shift(1)).loc[
            stk_signals.loc[
                (stk_signals["Signal"].shift(1) == "Buy") & (stk_signals["Regime"].shift(1) == 1)].index
        ].tolist(),
        "End Date": stk_signals["Price"].loc[
            stk_signals.loc[
                (stk_signals["Signal"].shift(1) == "Buy") & (stk_signals["Regime"].shift(1) == 1)].index
        ].index
    })
    print(stk_long_profits)
    test_trade_periods(stk_long_profits, stk)
def test_trade_periods(stk_long_profits, stk):
    tradeperiods = pd.DataFrame({"Start": stk_long_profits.index,
                                 "End": stk_long_profits["End Date"]})
    stk_long_profits["Low"] = tradeperiods.apply(lambda x: min(stk.loc[x["Start"]:x["End"], "Adj. Low"]), axis=1)
    print(tradeperiods)
    cash = 1000000
    stk_backtest = pd.DataFrame({"Start Port. Value": [],
                                   "End Port. Value": [],
                                   "End Date": [],
                                   "Shares": [],
                                   "Share Price": [],
                                   "Trade Value": [],
                                   "Profit per Share": [],
                                   "Total Profit": [],
                                   "Stop-Loss Triggered": []})
    port_value = .1  # Max proportion of portfolio bet on any trade
    batch = 100  # Number of shares bought per batch
    stoploss = .2  # % of trade loss that would trigger a stoploss
    for index, row in stk_long_profits.iterrows():
        batches = np.floor(cash * port_value) // np.ceil(
            batch * row["Price"])  # Maximum number of batches of stocks invested in
        trade_val = batches * batch * row["Price"]  # How much money is put on the line with each trade
        if row["Low"] < (1 - stoploss) * row["Price"]:  # Account for the stop-loss
            share_profit = np.round((1 - stoploss) * row["Price"], 2)
            stop_trig = True
        else:
            share_profit = row["Profit"]
            stop_trig = False
        profit = share_profit * batches * batch  # Compute profits
        # Add a row to the backtest data frame containing the results of the trade
        stk_backtest = stk_backtest.append(pd.DataFrame({
            "Start Port. Value": cash,
            "End Port. Value": cash + profit,
            "End Date": row["End Date"],
            "Shares": batch * batches,
            "Share Price": row["Price"],
            "Trade Value": trade_val,
            "Profit per Share": share_profit,
            "Total Profit": profit,
            "Stop-Loss Triggered": stop_trig
        }, index=[index]))
        cash = max(0, cash + profit)
        print(stk_backtest)
        stk_backtest["End Port. Value"].plot()

def rolling_average_n_day(days, date_start, date_end, stk):
    """
    If the 200 day moving average is trending downward, overall bearish.
    If upward, bullish market.
    :param days:
    :param date_start:
    :param date_end:
    :param stk:
    :return:
    """
    for d in days:
        stk[str(d) + "d"] = np.round(stk["Adj. Close"].rolling(window=int(d), center=False).mean(), 2)
    #pandas_candlestick_ohlc(stk.loc[date_start:date_end, :], otherseries=[d + "d" for d in days], adj=True)

def benchmarking(spyderdat, start, end, bk):
    spyder = spyderdat.loc[start:end]
    batch = 100
    print(spyder.iloc[[0,-1],:])
    batches = np.ceil(100 * spyder.loc[:, "Adj Close"].iloc[0])  # Maximum number of batches of stocks invested in
    trade_val = batches * batch * spyder.loc[:,"Adj Close"].iloc[0] # How much money is used to buy SPY
    final_val = batches * batch * spyder.loc[:,"Adj Close"].iloc[-1] + (1000000 - trade_val) # Final value of the portfolio
    print(final_val)
    ax_bench = (spyder["Adj Close"] / spyder.loc[:, "Adj Close"].iloc[0]).plot(label="SPY")
    ax_bench = (bk["Portfolio Value"].groupby(level=0).apply(lambda x: x[-1]) / 1000000).plot(ax=ax_bench,                                                                         label="Portfolio")
    ax_bench.legend(ax_bench.get_lines(), [l.get_label() for l in ax_bench.get_lines()], loc='best')
    print(ax_bench)

"""
Following three functions taken from Source for educational purposes.
"""

def ma_crossover_orders(stocks, fast, slow):
    """
    :param stocks: A list of tuples, the first argument in each tuple being a string containing the ticker symbol of each stock (or however you want the stock represented, so long as it's unique), and the second being a pandas DataFrame containing the stocks, with a "Close" column and indexing by date (like the data frames returned by the Yahoo! Finance API)
    :param fast: Integer for the number of days used in the fast moving average
    :param slow: Integer for the number of days used in the slow moving average

    :return: pandas DataFrame containing stock orders

    This function takes a list of stocks and determines when each stock would be bought or sold depending on a moving average crossover strategy, returning a data frame with information about when the stocks in the portfolio are bought or sold according to the strategy
    """
    fast_str = str(fast) + 'd'
    slow_str = str(slow) + 'd'
    ma_diff_str = fast_str + '-' + slow_str

    trades = pd.DataFrame({"Price": [], "Regime": [], "Signal": []})
    for s in stocks:
        # Get the moving averages, both fast and slow, along with the difference in the moving averages
        s[1][fast_str] = np.round(s[1]["Close"].rolling(window=fast, center=False).mean(), 2)
        s[1][slow_str] = np.round(s[1]["Close"].rolling(window=slow, center=False).mean(), 2)
        s[1][ma_diff_str] = s[1][fast_str] - s[1][slow_str]

        # np.where() is a vectorized if-else function, where a condition is checked for each component of a vector, and the first argument passed is used when the condition holds, and the other passed if it does not
        s[1]["Regime"] = np.where(s[1][ma_diff_str] > 0, 1, 0)
        # We have 1's for bullish regimes and 0's for everything else. Below I replace bearish regimes's values with -1, and to maintain the rest of the vector, the second argument is apple["Regime"]
        s[1]["Regime"] = np.where(s[1][ma_diff_str] < 0, -1, s[1]["Regime"])
        # To ensure that all trades close out, I temporarily change the regime of the last row to 0
        regime_orig = s[1].loc[:, "Regime"].iloc[-1]
        s[1].loc[:, "Regime"].iloc[-1] = 0
        s[1]["Signal"] = np.sign(s[1]["Regime"] - s[1]["Regime"].shift(1))
        # Restore original regime data
        s[1].loc[:, "Regime"].iloc[-1] = regime_orig

        # Get signals
        signals = pd.concat([
            pd.DataFrame({"Price": s[1].loc[s[1]["Signal"] == 1, "Adj. Close"],
                          "Regime": s[1].loc[s[1]["Signal"] == 1, "Regime"],
                          "Signal": "Buy"}),
            pd.DataFrame({"Price": s[1].loc[s[1]["Signal"] == -1, "Adj. Close"],
                          "Regime": s[1].loc[s[1]["Signal"] == -1, "Regime"],
                          "Signal": "Sell"}),
        ])
        signals.index = pd.MultiIndex.from_product([signals.index, [s[0]]], names=["Date", "Symbol"])
        trades = trades.append(signals)

    trades.sort_index(inplace=True)
    trades.index = pd.MultiIndex.from_tuples(trades.index, names=["Date", "Symbol"])

    return trades
def pandas_candlestick_ohlc(dat, stick="day", adj=False, otherseries=None):
    """
    :param dat: pandas DataFrame object with datetime64 index, and float columns "Open", "High", "Low", and "Close",
        likely created via DataReader from "yahoo"
    :param stick: A string or number indicating the period of time covered by a single candlestick. Valid string inputs
        include "day", "week", "month", and "year", ("day" default), and any numeric input indicates the number of
        trading days included in a period
    :param adj: A boolean indicating whether to use adjusted prices
    :param otherseries: An iterable that will be coerced into a list, containing the columns of dat that hold other series to be plotted as lines

    This will show a Japanese candlestick plot for stock data stored in dat, also plotting other series if passed.
    """
    mondays = WeekdayLocator(MONDAY)  # major ticks on the mondays
    alldays = DayLocator()  # minor ticks on the days
    dayFormatter = DateFormatter('%d')  # e.g., 12

    # Create a new DataFrame which includes OHLC data for each period specified by stick input
    fields = ["Open", "High", "Low", "Close"]
    if adj:
        fields = ["Adj. " + s for s in fields]
    transdat = dat.loc[:, fields]
    transdat.columns = pd.Index(["Open", "High", "Low", "Close"])
    if (type(stick) == str):
        if stick == "day":
            plotdat = transdat
            stick = 1  # Used for plotting
        elif stick in ["week", "month", "year"]:
            if stick == "week":
                transdat["week"] = pd.to_datetime(transdat.index).map(lambda x: x.isocalendar()[1])  # Identify weeks
            elif stick == "month":
                transdat["month"] = pd.to_datetime(transdat.index).map(lambda x: x.month)  # Identify months
            transdat["year"] = pd.to_datetime(transdat.index).map(lambda x: x.isocalendar()[0])  # Identify years
            grouped = transdat.groupby(list(set(["year", stick])))  # Group by year and other appropriate variable
            plotdat = pd.DataFrame({"Open": [], "High": [], "Low": [],
                                    "Close": []})  # Create empty data frame containing what will be plotted
            for name, group in grouped:
                plotdat = plotdat.append(pd.DataFrame({"Open": group.iloc[0, 0],
                                                       "High": max(group.High),
                                                       "Low": min(group.Low),
                                                       "Close": group.iloc[-1, 3]},
                                                      index=[group.index[0]]))
            if stick == "week":
                stick = 5
            elif stick == "month":
                stick = 30
            elif stick == "year":
                stick = 365

    elif (type(stick) == int and stick >= 1):
        transdat["stick"] = [np.floor(i / stick) for i in range(len(transdat.index))]
        grouped = transdat.groupby("stick")
        plotdat = pd.DataFrame(
            {"Open": [], "High": [], "Low": [], "Close": []})  # Create empty data frame containing what will be plotted
        for name, group in grouped:
            plotdat = plotdat.append(pd.DataFrame({"Open": group.iloc[0, 0],
                                                   "High": max(group.High),
                                                   "Low": min(group.Low),
                                                   "Close": group.iloc[-1, 3]},
                                                  index=[group.index[0]]))

    else:
        raise ValueError(
            'Valid inputs to argument "stick" include the strings "day", "week", "month", "year", or a positive integer')

    # Set plot parameters, including the axis object ax used for plotting
    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.2)
    if plotdat.index[-1] - plotdat.index[0] < pd.Timedelta('730 days'):
        weekFormatter = DateFormatter('%b %d')  # e.g., Jan 12
        ax.xaxis.set_major_locator(mondays)
        ax.xaxis.set_minor_locator(alldays)
    else:
        weekFormatter = DateFormatter('%b %d, %Y')
    ax.xaxis.set_major_formatter(weekFormatter)

    ax.grid(True)

    # Create the candelstick chart
    candlestick_ohlc(ax, list(
        zip(list(dates.date2num(plotdat.index.tolist())), plotdat["Open"].tolist(), plotdat["High"].tolist(),
            plotdat["Low"].tolist(), plotdat["Close"].tolist())),
                     colorup="black", colordown="red", width=stick * .4)

    # Plot other series (such as moving averages) as lines
    if otherseries != None:
        if type(otherseries) != list:
            otherseries = [otherseries]
        dat.loc[:, otherseries].plot(ax=ax, lw=1.3, grid=True)

    ax.xaxis_date()
    ax.autoscale_view()
    plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')

    plt.show()
def backtest(signals, cash, port_value=.1, batch=100):
    """
    :param signals: pandas DataFrame containing buy and sell signals with stock prices and symbols, like that returned by ma_crossover_orders
    :param cash: integer for starting cash value
    :param port_value: maximum proportion of portfolio to risk on any single trade
    :param batch: Trading batch sizes

    :return: pandas DataFrame with backtesting results

    This function backtests strategies, with the signals generated by the strategies being passed in the signals DataFrame. A fictitious portfolio is simulated and the returns generated by this portfolio are reported.
    """

    SYMBOL = 1  # Constant for which element in index represents symbol
    portfolio = dict()  # Will contain how many stocks are in the portfolio for a given symbol
    port_prices = dict()  # Tracks old trade prices for determining profits
    # Dataframe that will contain backtesting report
    results = pd.DataFrame({"Start Cash": [],
                            "End Cash": [],
                            "Portfolio Value": [],
                            "Type": [],
                            "Shares": [],
                            "Share Price": [],
                            "Trade Value": [],
                            "Profit per Share": [],
                            "Total Profit": []})

    for index, row in signals.iterrows():
        # These first few lines are done for any trade
        shares = portfolio.setdefault(index[SYMBOL], 0)
        trade_val = 0
        batches = 0
        cash_change = row[
                          "Price"] * shares  # Shares could potentially be a positive or negative number (cash_change will be added in the end; negative shares indicate a short)
        portfolio[index[SYMBOL]] = 0  # For a given symbol, a position is effectively cleared

        old_price = port_prices.setdefault(index[SYMBOL], row["Price"])
        portfolio_val = 0
        for key, val in portfolio.items():
            portfolio_val += val * port_prices[key]

        if row["Signal"] == "Buy" and row["Regime"] == 1:  # Entering a long position
            batches = np.floor((portfolio_val + cash) * port_value) // np.ceil(
                batch * row["Price"])  # Maximum number of batches of stocks invested in
            trade_val = batches * batch * row["Price"]  # How much money is put on the line with each trade
            cash_change -= trade_val  # We are buying shares so cash will go down
            portfolio[index[SYMBOL]] = batches * batch  # Recording how many shares are currently invested in the stock
            port_prices[index[SYMBOL]] = row["Price"]  # Record price
            old_price = row["Price"]
        elif row["Signal"] == "Sell" and row["Regime"] == -1:  # Entering a short
            pass
            # Do nothing; can we provide a method for shorting the market?
        # else:
        # raise ValueError("I don't know what to do with signal " + row["Signal"])

        pprofit = row[
                      "Price"] - old_price  # Compute profit per share; old_price is set in such a way that entering a position results in a profit of zero

        # Update report
        results = results.append(pd.DataFrame({
            "Start Cash": cash,
            "End Cash": cash + cash_change,
            "Portfolio Value": cash + cash_change + portfolio_val + trade_val,
            "Type": row["Signal"],
            "Shares": batch * batches,
            "Share Price": row["Price"],
            "Trade Value": abs(cash_change),
            "Profit per Share": pprofit,
            "Total Profit": batches * batch * pprofit
        }, index=[index]))
        cash += cash_change  # Final change to cash balance

    results.sort_index(inplace=True)
    results.index = pd.MultiIndex.from_tuples(results.index, names=["Date", "Symbol"])

    return results

def main():
    year = int(input("Starting year for statistical analysis: "))
    start = datetime.datetime(year, 1, 1)
    end = datetime.date.today()
    wiki = "WIKI/"
    input_stock = input("Choose a stock from the list:\nMSFT, GOOG, FB, TWTR, NFLX, AMZN, YHOO, GE, QCOM, IBM, HPQ, AAPL\n")
    """
    possible_stocks = ["MSFT", "GOOG", "FB", "TWTR", "NFLX", "AMZN", "YHOO", "GE", "QCOM", "IBM", "HPQ", "AAPL"]
    (microsoft, google, facebook, twitter, netflix,
     amazon, yahoo, ge, qualcomm, ibm, hp, apple) = (quandl.get(wiki + s, start_date=start,
                                                         end_date=end, authtoken=api_key) for s in possible_stocks)
    stocks_dict = {"msft":microsoft, "goog":google, "fb":facebook, "twtr":twitter, "nflx":netflix, "amzn":amazon,
                       "yhoo":yahoo, "ge":ge, "qcom":qualcomm, "ibm":ibm, "hpq":hp, "aapl":apple}
    stock = stocks_dict[input_stock.lower()]
    """
    apple = quandl.get(wiki + input_stock.upper(), start_date=start,end_date=end, authtoken=api_key)
    print(apple)
    type(apple)
    pd.DataFrame(apple).head()
    apple.head()
    plt.rcParams['figure.figsize'] = (15,9)
    #apple["Adj. Close"].plot(grid = True)
    #pandas_candlestick_ohlc(apple, adj=True, stick="month")
    microsoft, google = (quandl.get(wiki+ s, start_date =start, end_date=end, authtoken=api_key) for s in ["MSFT", "GOOG"])
    stocks = pd.DataFrame({"AAPL": apple["Adj. Close"],
                           "MSFT": microsoft["Adj. Close"],
                           "GOOG": google["Adj. Close"]
                           })
    head = stocks.head()
    print(head)
    #stocks.plot(secondary_y = ["AAPL", "MSFT"], grid = True)
    # Plot return_t,0 = price_t/price_0
    stock_return = stocks.apply(lambda x: x/ x[0])
    stock_return.head() -1
    #stock_return.plot(grid = True).axhline(y=1, color="black",lw=2)
    #plot the log difference of the growth of a stock
    # change_t = log(price_t) - log(price_t-1)
    stock_change = stocks.apply(lambda x: np.log(x) - np.log(x.shift(1)))
    stock_change.head()
    #stock_change.plot(grid = True).axhline(y=0, color="black",lw=2)

    """
    Compare performance of stocks to the performance of the overall market. SPY 
    represents the SPDR S&p 500 exchange-traded mutual fund, which represents value
    in the market. Data retrieved from Yahoo! Finance
    """
    spyderdat = pd.read_csv("HistoricalQuotes.csv")
    spyderdat = pd.DataFrame(spyderdat.loc[:, ["open", "high", "low", "close", "close"]].iloc[1:].as_matrix(),
                             index=pd.DatetimeIndex(spyderdat.iloc[1:, 0]),
                             columns=["Open", "High", "Low", "Close", "Adj Close"]).sort_index()

    spyder = spyderdat.loc[start:end]

    stocks = stocks.join(spyder.loc[:, "Adj Close"]).rename(columns={"Adj Close": "SPY"})
    stocks.head()
    stock_return = stocks.apply(lambda x: x / x[0])
    #stock_return.plot(grid=True).axhline(y=1, color="black", lw=2)

    stock_change = stocks.apply(lambda x: np.log(x) - np.log(x.shift(1)))
    #stock_return.plot(grid=True).axhline(y=0, color="black", lw=2)

    """
    Annualize our returns by computing the annual percentage rate
    """
    stock_change_apr = stock_change * 252 * 100
    stock_change_apr.tail()
    """
    Get the risk-free rate (rate of return on a risk-free financial asset.
    """
    tbill = quandl.get("FRED/TB3MS", start_date=start,end_date=end, authtoken=api_key)
    tbill.tail()
    #tbill.plot()
    # get the most recent treasury bill rate
    rrf = tbill.iloc[-1,0]
    """
    Make a linear regression model of the form: y_i = alpha + betax_i
    How much a stock moves in relation to the market: 
        beta = r(s_y/s_x)
        If beta > 0, stock generally moves in direction of market,
        If beta == 1, stock moves strongly in response to the market
        If |beta| < 1, the stock is less responsive to the market
    The average excess return over the market: alpha = y - betax
    The return of a financial asset (R_t) = alpha + beta(R_Mt - r_RF) + e_t + r_RF
    R_t - r_RF is the excess return (return exceeding risk-free rate of return
    R_Mt is the return of the market at time t
    """
    # get series that contains how much each stock is correlated with SPY
    smcorr = stock_change_apr.drop("SPY", 1).corrwith(
        stock_change_apr.SPY)  # Since RRF is constant it doesn't change the
    sy = stock_change_apr.drop("SPY", 1).std()
    sx= stock_change_apr.SPY.std()
    ybar = stock_change_apr.drop("SPY",1).mean() - rrf
    xbar = stock_change_apr.SPY.mean() - rrf
    beta = smcorr * sy / sx
    alpha = ybar - beta * xbar
    """
    Calculate Shape ratio = (R_t - r_RF)/s where s is the volatility of the stock 
    """
    sharpe = (ybar -rrf) / sy
    """
    Find trends in stocks with a q-day moving average
    """
    #rolling_average_n_day(["20"], '2016-01-04', '2016-12-31', apple)
    start = datetime.datetime(2010, 1, 1)
    apple = quandl.get("WIKI/AAPL", start_date=start, end_date=end, authtoken=api_key)
    rolling_average_n_day(["20", "50", "200"], '2016-01-04', '2016-12-31', apple)
    moving_average_crossover_strategy(["20", "50"], apple, '2016-01-04', '2016-12-31')
    """
    signals = ma_crossover_orders([("AAPL", apple),
                                   ("MSFT", microsoft),
                                   ("GOOG", google),
                                   ("FB", facebook),
                                   ("TWTR", twitter),
                                   ("NFLX", netflix),
                                   ("AMZN", amazon),
                                   ("YHOO", yahoo),
                                   ("GE", ge),
                                   ("QCOM", qualcomm),
                                   ("IBM", ibm),
                                   ("HPQ", hp)],
                                  fast=20, slow=50)

    #print(signals)
    bk = backtest(signals, 1000000)
    bk["Portfolio Value"].groupby(level=0).apply(lambda x: x[-1]).plot()
    benchmarking(spyderdat, start, end, bk)
                                      """
    plt.show()


main()
