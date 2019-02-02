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

def rolling_average_n_day(days, date_start, date_end, stk, stock_name):
    """
    Calculate the rolling average over n days, where the average is calculated by
    (1/n) summation from (i=0) to (n-1) of (x_t-i)
    This moving average helps to identify trends from the noise.
    If the 200 day moving average is trending downward, overall bearish.
    If upward, bullish market.
    :param days: An array of n elements to calculate an n-day moving average from
    :param date_start: The date to start from
    :param date_end: The end date
    :param stk: The stock to calculate the average for
    :return: None
    """
    for d in days:
        stk[str(d) + "d"] = np.round(stk["Adj. Close"].rolling(window=int(d), center=False).mean(), 2)
    pandas_candlestick_ohlc(stk.loc[date_start:date_end, :], stock_name, otherseries=[d + "d" for d in days], adj=True)

def calculate_returns(stocks):
    """
    Calculate and plot the stock's return since the beginning of the period of interest, defined by
    return_t,0 = price_t / price_0
    Calculate the change of each stock per day, where the change is defined by
    change_t = log(price_t) - log(price_(t-1))
    :param stocks: The stocks to compare
    :return: none
    """
    # stock_return = price_t / price_0
    stock_return = stocks.apply(lambda x: x / x[0])
    print(stock_return.head() - 1)
    stock_return.plot(grid = True).axhline(y=1, color="black",lw=2)
    plt.title("Return since Beginning of Period of Interest")

    # plot the log difference of the growth of a stock
    # change_t = log(price_t) - log(price_(t-1))
    stock_change = stocks.apply(lambda x: np.log(x) - np.log(x.shift(1)))
    print(stock_change.head())
    stock_change.plot(grid = True).axhline(y=0, color="black",lw=2)
    plt.title("Growth of stocks")

def compare_performance_to_market(stocks, start, end):
    """
    Compare performance of stocks to the performance of the overall market. SPY
    represents the SPDR S&p 500 exchange-traded mutual fund, which represents value
    in the market. Data retrieved from Yahoo! Finance

    :param stocks: The stocks to compare performance to market
    :param start: The starting date chosen by user
    :param end: Today
    :return:
        spyderdat, a dataframe of the SPY data including open, high, low, and close
        stock_change, a dataframe of changes in the stocks vs SPY
    """
    spyderdat = pd.read_csv("HistoricalQuotes.csv")
    spyderdat = pd.DataFrame(spyderdat.loc[:, ["open", "high", "low", "close", "close"]].iloc[1:].as_matrix(),
                             index=pd.DatetimeIndex(spyderdat.iloc[1:, 0]),
                             columns=["Open", "High", "Low", "Close", "Adj Close"]).sort_index()

    spyder = spyderdat.loc[start:end]

    stocks = stocks.join(spyder.loc[:, "Adj Close"]).rename(columns={"Adj Close": "SPY"})
    print(stocks.head())

    print("Calculating the rate of return since the beginning of the period of interest compared to SPY...")
    stock_return = stocks.apply(lambda x: x / x[0])
    stock_return.plot(grid=True).axhline(y=1, color="black", lw=2)
    plt.title("Rate of Return since Beginning of Interest Period vs SPY")

    print("Calculating the growth of stocks compared to SPY")
    stock_change = stocks.apply(lambda x: np.log(x) - np.log(x.shift(1)))
    stock_change.plot(grid=True).axhline(y=0, color="black", lw=2)
    plt.title("Growth of Stocks vs SPY")

    return spyderdat, stock_change

def calculate_risk_metrics(stock_change, start, end):
    """
    Annualize our returns by computing the annual percentage rate
    :param stock_change: a dataframe tracking stock changes vs the SPY
    :param start: The start date specified by user
    :param end: Today
    :return:
        stock_change_apr: the annual percentage rate
        rrf: The risk free rate of return
    """
    stock_change_apr = stock_change * 252 * 100
    print(stock_change_apr.tail())

    # Get the risk-free rate (rate of return on a risk-free financial asset.
    print("Getting the risk-free rate of return, calculated by the yields of 3 month US Treasury Bills.")
    tbill = quandl.get("FRED/TB3MS", start_date=start,end_date=end, authtoken=api_key)
    print(tbill.tail())
    tbill.plot()
    plt.title("Risk Free Rate of Return (Obtained from 3-month US Treasury Bills)")

    # Get the most recent treasury bill rate
    rrf = tbill.iloc[-1,0]
    print("Most recent Treasury Bill rate: ")
    print(rrf)

    return stock_change_apr, rrf

def linear_regression_model(stock_change_apr, rrf, start, end):
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
    :param stock_change_apr: a dataframe for the annual percentage rate of the chosen stock
    :param rrf: The risk-free rate of return
    :param start: The start date
    :param end: The end date
    :return: None
    """
    # get series that contains how much each stock is correlated with SPY
    smcorr = stock_change_apr.drop("SPY", 1).corrwith(
        stock_change_apr.SPY)  # Since RRF is constant it doesn't change
    print(smcorr)

    sy = stock_change_apr.drop("SPY", 1).std()
    sx= stock_change_apr.SPY.std()

    ybar = stock_change_apr.drop("SPY",1).mean() - rrf
    xbar = stock_change_apr.SPY.mean() - rrf

    beta = smcorr * sy / sx
    alpha = ybar - beta * xbar

    # Calculate Sharpe ratio = (R_t - r_RF)/s where s is the volatility of the stock
    sharpe = (ybar -rrf) / sy
    print("Sharpe ratio: ")
    print(sharpe)

def moving_average_crossover_strategy(days, stk, start_date, end_date):
    """
    Takes two moving averages, a 'fast' one and a 'slow one and calculates s strategy.
    Trade the asset when the fast moving average crosses over the slow moving average.
    Exit the trade when the fast moving average crosses over the slow moving average again.

    :param days: An array of the day ranges to calculate a moving average from. Should have
        two elements, the fast and slow
    :param stk: The particular stock to examine
    :param start_date: The starting date
    :param end_date: The end date
    :return: The stock
    """
    ds = [d + "d" for d in days]
    start_range = ds[0]
    end_range = ds[1]
    stk[start_range] = np.round(stk["Adj. Close"].rolling(window=int(days[0]), center=False).mean(), 2)
    stk[end_range] = np.round(stk["Adj. Close"].rolling(window=int(days[1]), center=False).mean(), 2)

    range = (start_range+"-"+end_range)
    stk[range] = stk[start_range] - stk[end_range]
    print(stk.tail())

    stk["Regime"] = np.where(stk[range] > 0,1,0)

    # We have 1's for bullish regimes and 0's for everything else
    # replace bearish regimes's values with -1, and to maintain the rest of the vector,
    # the second argument is stk["Regime"]
    stk["Regime"] = np.where(stk[range] < 0,-1, stk["Regime"])
    stk.loc[start_date:end_date,"Regime"].plot(ylim=(-2,2)).axhline(y=0, color="black",lw=2)
    print(stk.tail())


    # stk["Regime"].plot(ylim=(-2, 2)).axhline(y=0, color="black", lw=2)
    count = stk["Regime"].value_counts().tolist()
    print("The market was bearish for " + str(count[0]) + " days.")
    print("The market was bullish for " + str(count[1]) + " days.")
    print("The market was neutral for " + str(count[2]) + " days.")

def get_signals(stk):
    """
    Obtain the signals of whether to sell or buy, calculated by:
        s_t = sign(r_t - r_(t-1)) where s_t is an element of {-1,0,1} where
        -1 indicates sell
        1 indicates buy
        0 indicates no action
    :param stk: The stock to calculate signals for
    :return: none
    """
    # To ensure that all trades close out, temporarily change the regime of the last row to 0
    regime_orig = stk.loc[:,"Regime"].iloc[-1]
    stk.loc[:, "Regime"].iloc[-1] = 0
    stk["Signal"] = np.sign(stk["Regime"] - stk["Regime"].shift(1))

    # restore data
    stk.loc[:, "Regime"].iloc[-1] = regime_orig
    print(stk.tail())

    #stk["Signal"].plot(ylim=(-2,2))
    count = stk["Signal"].value_counts().tolist()

    print("We would buy stock " + str(count[2]) + " times.")
    print("We would sell stock " + str(count[1]) + " times.")
    print("We do nothing " + str(count[0]) + " times.")
    identify_prices_at_close(stk)

def identify_prices_at_close(stk):
    """

    :param stk:
    :return:
    """
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
#    long_trade_profitability(stk_signals, stk)

def long_trade_profitability(stk_signals, stk):
    """

    :param stk_signals:
    :param stk:
    :return:
    """
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
    #test_trade_periods(stk_long_profits, stk)

def test_trade_periods(stk_long_profits, stk):
    """

    :param stk_long_profits:
    :param stk:
    :return:
    """
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

def benchmarking(spyderdat, start, end, bk):
    """

    :param spyderdat:
    :param start:
    :param end:
    :param bk:
    :return:
    """
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
Following three functions taken from Source - some modifications done for
educational purposes. 
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
def pandas_candlestick_ohlc(dat, stock_name, stick="day", adj=False, otherseries=None):
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
    plt.title(stock_name)

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
    """
    Main function to calculate various trends, strategies, and plot results for stocks.
    """
    year = int(input("Starting year for statistical analysis: "))
    input_stock = input("Choose a stock:\n(EX: AAPL or FB)\n").upper()
    plot = input("Type Y to plot " + input_stock + ". Press enter to skip.")

    start = datetime.datetime(year, 1, 1)
    end = datetime.date.today()
    wiki = "WIKI/"
    stock = quandl.get(wiki + input_stock, start_date=start,end_date=end, authtoken=api_key)

    type(stock)
    pd.DataFrame(stock).head()
    # print the values of the first part of the stock dataframe
    print(stock.head())

    # Upon user request, plot the chosen stock's Adj. Close as well as a candlestick plot per month.
    if plot.lower() == "y":
        stock["Adj. Close"].plot(grid = True)
        plt.title("Adj. Close for " + input_stock)
        pandas_candlestick_ohlc(stock, input_stock, adj=True, stick="month")

    # Get some comparison stocks
    print("Getting stocks for Apple, Microsoft, and Google for comparison...")
    apple, microsoft, google = (quandl.get(wiki + s, start_date =start, end_date=end, authtoken=api_key) for s in
                                ["AAPL", "MSFT", "GOOG"])
    plot = input("Type Y to plot Apple, Microsoft, and Google stocks. Press enter to skip.")

    stocks = pd.DataFrame({"AAPL": apple["Adj. Close"],
                           "MSFT": microsoft["Adj. Close"],
                           "GOOG": google["Adj. Close"],
                           input_stock:stock["Adj. Close"]
                           })
    # print the first values of the dataframe for all of the stocks.
    print(stocks.head())

    # if user desires, plot the four stocks
    if plot.lower() == "y":
        stocks.plot(secondary_y = ["AAPL", "MSFT", "GOOG", input_stock], grid = True)
        plt.title("AAPL, MSFT, GOOG, and " + input_stock)

    # Plot rate of return for chosen stock
    plot = input("Type Y to plot rate of return for " + input_stock + ". Press enter to skip.")
    if plot.lower() == "y":
        calculate_returns(stocks)

    # Calculate the performance of the stocks versus the market average
    print("Comparing performance of stocks to performance of overall market using SPY.")
    spyderdat, stock_change = compare_performance_to_market(stocks, start, end)

    # Calculate risk metrics to make a linear regression model
    print("Calculating risk metrics")
    stock_change_apr, rrf = calculate_risk_metrics(stock_change, start, end)

    # Create a linear regression model
    print("Creating a linear regression model")
    linear_regression_model(stock_change_apr, rrf, start, end)

    # Find trends in stocks with a n-day moving average
    print("Calculating a n-day moving average")
    n_day = input("Enter a value for n. Enter 0 to quit")
    while True:
        if not n_day.isdigit():
            print("Sorry, could not understand that input. Enter 0 to quit or try again.")
            n_day = input("Enter a value for n. Enter 0 to quit")
        if int(n_day) == 0:
            break
        rolling_average_n_day([n_day], str(year)+'-01-04', str(year)+'-12-31', stock, input_stock)
        n_day = input("Enter a value for q. Enter 0 to quit")

    # Calculate a 20d moving average to compare
    print("Calculating a 20d moving average since 2010 for comparison purposes.")
    start = datetime.datetime(2010, 1, 1)
    stock = quandl.get("WIKI/" + input_stock, start_date=start, end_date=end, authtoken=api_key)
    rolling_average_n_day(["20"], str(year) + '-01-04', str(year) + '-12-31', stock, input_stock)
    pandas_candlestick_ohlc(stock.loc[str(year)+'-01-04':str(year)+'-12-31', :],input_stock, otherseries="20d", adj=True)

    # Ask user for a range of moving averages to calculate
    print("Calculate multiple moving averages.")
    q_day_range = input("Enter a list of comma and space separated q_days (Ex: 20, 50, 200). Enter 0 to quit:\n")
    while True:
        split_range = q_day_range.split(", ")
        if not split_range[0].isdigit() or int(split_range[0]) == 0:
            break
        rolling_average_n_day(split_range, str(year)+'-01-04', str(year)+'-12-31', stock, input_stock)
        q_day_range = input("Enter a list of comma and space separated q_days (Ex: 20, 50, 200). Enter 0 to quit:\n")

    # Calculate a moving average crossover strategy with fast and slow values chosen by user
    print("Moving average crossover strategy:")
    slow = input("Select a slow moving average of days (Ex: 20):\n")
    fast = input("Select a fast moving average of days (Ex: 50):\n")
    moving_average_crossover_strategy([slow, fast], stock, str(year)+'-01-04', str(year)+'-12-31')

    # Get the signals of when to buy, trade, or sell
    print("Getting signals for when to buy, trade, or sell.")
    get_signals(stock)

    # test functions commented out below.
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
