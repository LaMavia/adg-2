import altair as alt
import pandas as pd
import sys

def main():
    path = sys.argv[1]
    data = pd.read_csv(path, delimiter=";")

    w = 500

    error_bars = alt.Chart(data).mark_errorbar(extent='stdev').encode(
        y=alt.Y('auc_roc').scale(zero=False),
        x=alt.X('rate')
    )

    points = alt.Chart(data, width=w).mark_point(filled=True, color='black').encode(
        y=alt.Y('mean(auc_roc)'),
        x=alt.X('rate'),
    )

    chart = error_bars + points

    # chart = alt.Chart(data, width=100).transform_density(
    #     'auc_roc',
    #     as_=['auc_roc', 'density'],
    #     groupby=['rate']
    # ).mark_area(orient='horizontal').encode(
    #     alt.X('density:Q')
    #         .stack('center')
    #         .impute(None)
    #         .title(None)
    #         .axis(labels=False, values=[0], grid=False, ticks=True),
    #     alt.Y('auc_roc:Q'),
    #     alt.Color('rate:N'),
    #     alt.Column('rate:N')
    #         .spacing(0)
    #         .header(titleOrient='bottom', labelOrient='bottom', labelPadding=0)
    # ).configure_view(
    #     stroke=None
    # )

    chart.save(sys.argv[2]) # sample_test.html

if __name__ == "__main__":
    main()
