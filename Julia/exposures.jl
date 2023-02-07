using DataFrames
using CSV
using Dates
using BenchmarkTools
using ExperienceAnalysis
using DayCounts

function expsoures_ExperienceAnalysis(
    df_yearly::DataFrame,
    study_start::Date,
    study_end::Date,
)
    continue_exposure = df_yearly.status .== "Surrender"
    to = [ismissing(d) ? study_end : min(study_end, d) for d in df_yearly.term_date]
    df_yearly.exposure =
        exposure.(
            ExperienceAnalysis.Anniversary(Year(1)),   # The basis for our exposures
            df_yearly.issue_date,                             # The `from` date
            to,                                    # the `to` date array we created above
            continue_exposure,
        )
    df_yearly = flatten(df_yearly, :exposure)
    df_yearly = filter(row -> row.exposure.to >= study_start, df_yearly)
    df_yearly.exposure =
        map(e -> (from = max(study_start, e.from), to = e.to), df_yearly.exposure)
    df_yearly.exposure_fraction =
        map(e -> yearfrac(e.from, e.to, DayCounts.Thirty360()), df_yearly.exposure)
    return df_yearly
end

function run_exposure_benchmarks()
    df = CSV.read("../data/census_dat.csv", DataFrame)
    df.term_date = [d == "NA" ? missing : Date(d, "yyyy-mm-dd") for d in df.term_date]
    study_end = Date(2020, 2, 29)
    study_start = Date(2006, 6, 15)
    df_yearly = copy(df)
    result = expsoures_ExperienceAnalysis(df_yearly, study_start, study_end)
    b = @benchmark expsoures_ExperienceAnalysis($df_yearly, $study_start, $study_end)
    return Dict(
        "ExperienceAnalysis" => Dict(
            "num_rows" => size(result, 1),
            "mean" => string(mean(b)),
        ),
    )
end