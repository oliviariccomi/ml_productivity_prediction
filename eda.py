import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

#     plt.savefig("./images/productivity_team_dep.png")

show_plots = True

#########################
#   USEFUL FUNCTIONS    #
#########################

def incentive_outliers(df):
    q1 = df["incentive"].quantile(0.25)
    q3 = df["incentive"].quantile(0.75)
    iqr = q3 - q1
    ww = 1.5
    lower = q1 - (ww * iqr)
    upper = q3 + (ww * iqr)

    df['incentive'] = np.where(df["incentive"] > upper + 20, upper,
                               np.where(df["incentive"] < lower, lower, df["incentive"]))
    return df

def show_values_on_bars(axes, n_decimal):
    def _show_on_single_plot(graph):
        for p in graph.patches:
            _x = p.get_x() + p.get_width() / 2
            _y = p.get_y() + p.get_height()
            if n_decimal == 3:
                value = '{:.3f}'.format(p.get_height())
            else:
                value = '{:.0f}'.format(p.get_height())
            graph.text(_x, _y+20, value, va='center_baseline', ha="center")

    if isinstance(axes, np.ndarray):
        for idx, a in np.ndenumerate(axes):
            _show_on_single_plot(a)
    else:
        _show_on_single_plot(axes)



#################
#    PLOTS      #
#################

# Target variable

def productivityDistribution(df):
    sns.histplot(data=df, x='actual_productivity', palette="rocket")
    plt.xlabel("Actual Productivity")
    plt.show()


def targeted_actual(df):
    df['diff'] = df.actual_productivity - df.targeted_productivity

    df['Target_label'] = np.nan

    df.loc[df['diff'] < 0, 'Target_label'] = -1
    df.loc[(df['diff'] == 0), 'Target_label'] = 0
    df.loc[df['diff'] > 0, 'Target_label'] = 1
    g = sns.countplot(x='Target_label', data=df, palette="Blues_d")
    show_values_on_bars(g, 0)
    plt.show()

    df.drop(df[['Target_label', 'diff']], axis=1, inplace=True)

# Department vs actual productivity
def department_productivity(df):
    sns.kdeplot(data=df, x="actual_productivity", hue="department", palette="rocket", fill=True)
    plt.xlabel("Actual Productivity")
    plt.show()


# Department vs targeted
def department_targeted(df):
    sns.kdeplot(data=df, x="targeted_productivity", hue="department", palette="rocket")
    plt.ylabel("Targeted Productivity")
    plt.show()


# Quarter vs Day
def quarter_day(df):
    ax = sns.lineplot(data=df, x="day", y="quarter", palette="rocket")
    ax.set_yticks(range(0, 5, 1))
    plt.xlabel("Day")
    plt.ylabel("Quarter")
    plt.show()


# Number of Workers vs SVM
def workers_smv(df):
    sns.lineplot(data=df, x="smv", y="no_of_workers", palette="rocket")
    plt.xlabel("SMV")
    plt.ylabel("Number of Workers")
    plt.show()


# Number of Workers vs Department
def workers_department(df):
    data = df.groupby(['department']).no_of_workers.sum()
    data.plot.pie(autopct='%1.1f%%')
    plt.ylabel(None)
    plt.show()


def prod_team_dep(df):
    plt.figure(figsize=(15, 6))
    sns.barplot(data=df, x='team', y='actual_productivity', hue='department')
    plt.ylabel("Actual Productivity")
    plt.xlabel("Team")
    plt.show()


def prod_month_incentive(df):
    df['group_incentive'] = np.where(
        (df['incentive'] >= 0) & (df['incentive'] <= 20), "0-20",
        np.where((df['incentive'] >= 21) & (df['incentive'] <= 40), "21-40",
                 np.where((df['incentive'] >= 41) & (df['incentive'] <= 60), "41-60",
                          np.where((df['incentive'] >= 61) & (df['incentive'] <= 80), "61-80",
                                   np.where((df['incentive'] >= 81) & (df['incentive'] <= 100), "81-100",
                                            np.where((df['incentive'] > 100), "more than 100",
                                                     0))
                                   ))))

    plt.figure(figsize=(10, 7))
    sns.barplot(data=df, x='month', y='actual_productivity', hue='group_incentive', palette="Blues_d", ci=None)
    plt.legend(title="Incentives", bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
    plt.subplots_adjust(left=.07, right=.83)
    plt.ylabel("Actual Productivity")
    plt.xlabel("Month")
    plt.xticks([0, 1, 2], ['January', 'February', 'March'])
    plt.show()

    df.drop(df[['group_incentive']], axis=1, inplace=True)

def prod_department_incentive(df):
    df['group_incentive'] = np.where(
        (df['incentive'] >= 0) & (df['incentive'] <= 20), "0-20",
        np.where((df['incentive'] >= 21) & (df['incentive'] <= 40), "21-40",
                 np.where((df['incentive'] >= 41) & (df['incentive'] <= 60), "41-60",
                          np.where((df['incentive'] >= 61) & (df['incentive'] <= 80), "61-80",
                                   np.where((df['incentive'] >= 81) & (df['incentive'] <= 100), "81-100",
                                            np.where((df['incentive'] > 100), "more than 100",
                                                     0))
                                   ))))

    plt.figure(figsize=(10, 7))
    sns.barplot(data=df, x='department', y='actual_productivity', hue='group_incentive', palette="rocket", ci=None)
    plt.legend(title="Incentives in BDT", bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
    plt.subplots_adjust(left=.07, right=.83)
    plt.ylabel("Actual Productivity")
    plt.xlabel("Department")
    plt.show()

    df.drop(df[['group_incentive']], axis=1, inplace=True)



def overtime_incentive(df):
    df['group_incentive'] = np.where(
        (df['incentive'] >= 0) & (df['incentive'] <= 20), "0-20",
        np.where((df['incentive'] >= 21) & (df['incentive'] <= 40), "21-40",
                 np.where((df['incentive'] >= 41) & (df['incentive'] <= 60), "41-60",
                          np.where((df['incentive'] >= 61) & (df['incentive'] <= 80), "61-80",
                                   np.where((df['incentive'] >= 81) & (df['incentive'] <= 100), "81-100",
                                            np.where((df['incentive'] > 100), "more than 100",
                                                     0))
                                   ))))
    sns.violinplot(data=df, x='group_incentive', y='over_time', palette="Blues_d")
    plt.xlabel("Incentives in BDT")
    plt.ylabel("Over Time")
    plt.show()

    df.drop(df[['group_incentive']], axis=1, inplace=True)


def overtime_team_department(df):
    plt.figure(figsize=(8, 6))
    sns.barplot(data=df, x='team', y='over_time', palette="rocket", hue='department', ci=None)
    plt.subplots_adjust(left=.09, right=.84)
    plt.legend(title="department", bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
    plt.xlabel("Team")
    plt.ylabel("Over Time")
    plt.show()


def incentives_ouliers(df):
    sns.boxplot(data=df, x='incentive', palette="rocket")
    plt.xlabel("Incentive")
    plt.show()

    q1 = df["incentive"].quantile(0.25)
    q3 = df["incentive"].quantile(0.75)
    iqr = q3 - q1
    ww = 1.5
    lower = q1 - (ww * iqr)
    upper = q3 + (ww * iqr)

    var = df.loc[df["incentive"] > upper, ["incentive", "actual_productivity"]]
    fig, ax = plt.subplots()
    sns.barplot(x=var["incentive"], y=var["actual_productivity"], palette="rocket")
    ax.axhline(y=df["actual_productivity"].mean(), linewidth=1, color="red", label="Productivity Mean")
    plt.legend()
    plt.xlabel("Incentive")
    plt.ylabel("Actual Productivity")
    plt.show()
    print(df["actual_productivity"].mean())

def overtime_productivity(df):
    sns.scatterplot(data=df, x="over_time", y="actual_productivity", color="#1f78b4")
    plt.xlabel("Over Time")
    plt.ylabel("Actual Productivity")
    plt.show()

def day_productivity(df):
    sns.barplot(data=df, y="actual_productivity", x="week_day", palette="rocket", ci=None)
    plt.xlabel("Day of the week")
    plt.ylabel("Actual Productivity")
    plt.show()

# Heatmaps
def heatmap(df):
    plt.figure(figsize=(7, 7))
    upp_mat = np.triu(df.corr())
    sns.heatmap(df.corr(), vmin=-1, vmax=+1, annot=True, cmap="rocket", mask=upp_mat, square=True)
    plt.show()


def ordered_matrix(df):
    corr_matrix = df.corr()
    k = 17
    cols = corr_matrix.nlargest(k, "actual_productivity")["actual_productivity"].index
    d = df.reindex(columns=cols)
    plt.figure(figsize=(15, 10))
    new = d.corr()
    m = np.triu(new, k=1)
    a = sns.heatmap(new, annot=True, annot_kws={"fontsize": 9}, mask=m)
    a.set_xticklabels(a.get_xmajorticklabels(), fontsize=12)
    a.set_yticklabels(a.get_ymajorticklabels(), fontsize=13)
    cbar = a.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)
    plt.subplots_adjust(top=.9, left=.15, bottom=.2, right=.99)
    plt.show()


def eda_analysis(df):
    ## First: STATISTICS
    print(df.info())
    print(df.describe())

    # Percentage of null variables
    print(df.isnull().sum() * 100 / len(df))

    # Dataset dimensions
    print(df.shape)

    ## A little bit of preprocessing in order to plot correct functions

    df['department'] = df['department'].apply(
        lambda x: 'finishing' if (x == 'finishing ' or x == 'finishing') else 'sweing')
    df = df.rename(columns={'day': 'week_day'})

    # Handling datetime
    df['day'] = df['date'].apply(lambda x: int(x.split("/")[1]))
    df['month'] = df['date'].apply(lambda x: int(x.split("/")[0]))
    df['year'] = df['date'].apply(lambda x: int(x.split("/")[2]))

    df.drop(['date'], axis=1, inplace=True)  # dropping date time after transformation

    # Filling Nan values (for now...)
    df['wip'].fillna(int(df['wip'].mean()), inplace=True)

    incentives_ouliers(df)
    # Fixing incentive outliers
    incentive_outliers(df)

    # Main graphics

    if show_plots:
        d = ['#a6cee3', "#1f78b4"]
        sns.set_palette(sns.color_palette(d))

        productivityDistribution(df)
        targeted_actual(df)
        workers_smv(df)
        department_productivity(df)
        workers_department(df)
        prod_month_incentive(df)
        prod_department_incentive(df)
        overtime_incentive(df)
        overtime_team_department(df)
        overtime_productivity(df)
        day_productivity(df)

    dummies = pd.get_dummies(df.department)
    df = pd.concat([df, dummies], axis='columns')
    df.drop(['department'], axis=1, inplace=True)

    le_quarter = LabelEncoder()
    df['quarter'] = le_quarter.fit_transform(df['quarter'])
    le_quarter_name_mapping = dict(zip(le_quarter.classes_, le_quarter.transform(le_quarter.classes_)))
    # print(le_name_mapping)

    le_day = LabelEncoder()
    df['week_day'] = le_day.fit_transform(df['week_day'])
    le_day_name_mapping = dict(zip(le_day.classes_, le_day.transform(le_day.classes_)))
    # print(le_day_name_mapping)

    if show_plots:
        # Heatmap
        ordered_matrix(df)
    return df
