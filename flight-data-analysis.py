from pyspark.sql import SparkSession
from pyspark.sql.functions import row_number, col, abs, unix_timestamp, when, stddev, count, sum, hour, avg
from pyspark.sql import Window
from pyspark.sql import functions as F

# Create a Spark session
spark = SparkSession.builder.appName("Advanced Flight Data Analysis").getOrCreate()

# Load datasets
flights_df = spark.read.csv("flights.csv", header=True, inferSchema=True)
airports_df = spark.read.csv("airports.csv", header=True, inferSchema=True)
carriers_df = spark.read.csv("carriers.csv", header=True, inferSchema=True)

# Define output paths
output_dir = "output/"
task1_output = output_dir + "task1_largest_discrepancy.csv"
task2_output = output_dir + "task2_consistent_airlines.csv"
task3_output = output_dir + "task3_canceled_routes.csv"
task4_output = output_dir + "task4_carrier_performance_time_of_day.csv"

def task1_largest_discrepancy(flights_df, carriers_df):
    # Calculate Scheduled and Actual Travel Time (in minutes)
    flights_df = flights_df.withColumn(
        "ScheduledTravelTime",
        (F.unix_timestamp("ScheduledArrival") - F.unix_timestamp("ScheduledDeparture")) / 60
    ).withColumn(
        "ActualTravelTime",
        (F.unix_timestamp("ActualArrival") - F.unix_timestamp("ActualDeparture")) / 60
    )

    # Calculate the absolute discrepancy between Scheduled and Actual Travel Time
    flights_df = flights_df.withColumn(
        "Discrepancy",
        F.abs(flights_df["ScheduledTravelTime"] - flights_df["ActualTravelTime"])
    )

    # Define a window for ranking flights by discrepancy within each carrier
    window_spec = Window.partitionBy("CarrierCode").orderBy(F.desc("Discrepancy"))

    # Rank flights by the discrepancy within each carrier
    flights_df = flights_df.withColumn("Rank", F.row_number().over(window_spec))

    # Rename CarrierCode in carriers_df to avoid ambiguity in join
    carriers_df = carriers_df.withColumnRenamed("CarrierCode", "CarrierCode_Carriers")

    # Join with carriers_df to include CarrierName and filter only the top-ranked flights per carrier
    flights_with_carrier = flights_df.join(
        carriers_df, flights_df.CarrierCode == carriers_df.CarrierCode_Carriers, "left"
    ).select(
        "FlightNum", flights_df["CarrierCode"], "CarrierName", "Origin", "Destination",
        "ScheduledTravelTime", "ActualTravelTime", "Discrepancy", "Rank"
    ).filter("Rank = 1")

    # Write the result to a CSV file with overwrite mode
    flights_with_carrier.write.mode("overwrite").csv(task1_output, header=True)
    print(f"Task 1 output written to {task1_output}")


# ------------------------
# Task 2: Most Consistently On-Time Airlines Using Standard Deviation
# ------------------------
def task2_consistent_airlines(flights_df, carriers_df):
    flights_df = flights_df.withColumn("DepartureDelay", unix_timestamp("ActualDeparture") - unix_timestamp("ScheduledDeparture"))

    consistent_airlines = flights_df.groupBy("CarrierCode") \
                                    .agg(count("FlightNum").alias("FlightCount"), stddev("DepartureDelay").alias("DelayStdDev")) \
                                    .filter(col("FlightCount") > 100) \
                                    .join(carriers_df, "CarrierCode") \
                                    .select("CarrierName", "FlightCount", "DelayStdDev") \
                                    .orderBy("DelayStdDev")

    consistent_airlines.write.mode("overwrite").csv(task2_output, header=True)

# # ------------------------
# # Task 3: Origin-Destination Pairs with the Highest Percentage of Canceled Flights
# # ------------------------
def task3_canceled_routes(flights_df, airports_df):
    # Define a canceled flight as one where ActualDeparture is null
    canceled_flights = flights_df.filter(F.col("ActualDeparture").isNull())

    # Calculate total flights for each origin-destination pair
    total_flights = flights_df.groupBy("Origin", "Destination").agg(F.count("*").alias("TotalFlights"))

    # Calculate canceled flights for each origin-destination pair
    canceled_flights_count = canceled_flights.groupBy("Origin", "Destination").agg(F.count("*").alias("CanceledFlights"))
    # Join the total flights with canceled flights
    cancellation_rates = total_flights.join(canceled_flights_count, on=["Origin", "Destination"], how="left") \
        .fillna(0) \
        .withColumn("CancellationRate", F.col("CanceledFlights") / F.col("TotalFlights"))

    # Join with airports to get airport names and cities for both Origin and Destination
    cancellation_rates = cancellation_rates \
        .join(airports_df.withColumnRenamed("AirportCode", "Origin").withColumnRenamed("AirportName", "OriginAirportName").withColumnRenamed("City", "OriginCity"), on="Origin") \
        .join(airports_df.withColumnRenamed("AirportCode", "Destination").withColumnRenamed("AirportName", "DestinationAirportName").withColumnRenamed("City", "DestinationCity"), on="Destination") \
        .select("Origin", "OriginAirportName", "OriginCity", "Destination", "DestinationAirportName", "DestinationCity", "CancellationRate") \
        .orderBy(F.col("CancellationRate").desc())

    # Write the result to a CSV file
    cancellation_rates.write.mode("overwrite").csv(task3_output, header=True)
    print(f"Task 3 output written to {task3_output}")

# # ------------------------
# # Task 4: Carrier Performance Based on Time of Day
# # ------------------------
def task4_carrier_performance_time_of_day(flights_df, carriers_df):
    flights_df = flights_df.withColumn("DepartureHour", hour("ScheduledDeparture")) \
                           .withColumn("TimeOfDay", when((col("DepartureHour") >= 6) & (col("DepartureHour") < 12), "Morning") \
                                               .when((col("DepartureHour") >= 12) & (col("DepartureHour") < 18), "Afternoon") \
                                               .when((col("DepartureHour") >= 18) & (col("DepartureHour") < 24), "Evening") \
                                               .otherwise("Night")) \
                           .withColumn("DepartureDelay", unix_timestamp("ActualDeparture") - unix_timestamp("ScheduledDeparture"))

    carrier_performance = flights_df.groupBy("CarrierCode", "TimeOfDay") \
                                    .agg(avg("DepartureDelay").alias("AvgDepartureDelay")) \
                                    .join(carriers_df, "CarrierCode") \
                                    .select("CarrierName", "TimeOfDay", "AvgDepartureDelay") \
                                    .orderBy("TimeOfDay", "AvgDepartureDelay")

    carrier_performance.write.mode("overwrite").csv(task4_output, header=True)


# ------------------------
# Call the functions for each task
# ------------------------
task1_largest_discrepancy(flights_df, carriers_df)
task2_consistent_airlines(flights_df, carriers_df)
task3_canceled_routes(flights_df, airports_df)
task4_carrier_performance_time_of_day(flights_df, carriers_df)

# Stop the Spark session
spark.stop()