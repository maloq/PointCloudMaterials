import cdsapi
import os
import argparse
import asyncio
from pathlib import Path

MAX_CONCURRENT_REQUESTS = 6  # Limit to 6 parallel downloads


def main(output_dir, start_year, end_year):
    os.makedirs(output_dir, exist_ok=True)

    variables = [
        "total_precipitation",
    ]
    years = [str(y) for y in range(start_year, end_year + 1)]
    # Months are needed for the CDS API request, but instead of looping over them
    # individually we will provide the **full list** in a single request so that
    # the server returns the whole year at once. This drastically reduces the
    # number of API calls (from 12 per year to 1 per year).
    months = [f"{m:02d}" for m in range(1, 13)]

    dataset = "derived-era5-single-levels-daily-statistics"
    base_request = {
        "product_type": "reanalysis",
        "day": [
            "01", "02", "03", "04", "05", "06", "07", "08", "09",
            "10", "11", "12", "13", "14", "15", "16", "17", "18",
            "19", "20", "21", "22", "23", "24", "25", "26", "27",
            "28", "29", "30", "31"
        ],
        "daily_statistic": "daily_mean",
        "time_zone": "utc+00:00",
        "frequency": "1_hourly",
        "area": [75, 10, 35, 180],
    }

    client = cdsapi.Client()
    tasks_info = []

    for year in years:
        for variable in variables:
            request = base_request.copy()
            request["variable"] = [variable]
            request["year"] = year
            # Request the full list of months so we get a complete year in one file
            request["month"] = months

            if variable == "total_precipitation":
                request["daily_statistic"] = "daily_sum"
            else:
                request["daily_statistic"] = "daily_mean"

            variable_dir = os.path.join(output_dir, variable)
            os.makedirs(variable_dir, exist_ok=True)
            # One file per variable-year (all 12 months contained inside)
            target_file = Path(variable_dir) / f"{variable}_{year}.grib"

            if target_file.exists():
                print(f"File {target_file} already exists, skipping.")
                continue

            print(f"Scheduling download for {variable} {year} (all months)")

            # Save the info needed to build tasks later (avoid creating semaphore outside loop)
            tasks_info.append((request, target_file))

    async def _runner():
        """Asynchronous driver that respects the concurrency limit."""
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        tasks = [
            _download_with_retries(semaphore, client, dataset, req, tgt)
            for req, tgt in tasks_info
        ]

        if tasks:
            await asyncio.gather(*tasks)
        else:
            print("Nothing to download – all requested files already exist.")

    # Kick off the asynchronous downloads in a dedicated event loop.
    asyncio.run(_runner())

    print("Download script finished.")


async def _download_with_retries(
    semaphore: asyncio.Semaphore,
    client: "cdsapi.Client",
    dataset: str,
    request: dict,
    target_file: Path,
    retries: int = 5,
    initial_delay: int = 10,
):
    """Download a single request using the CDS API inside a thread, honouring a retry
    mechanism and an asyncio semaphore to cap concurrency.

    Args:
        semaphore: The shared asyncio.Semaphore controlling concurrency.
        client: An instantiated ``cdsapi.Client``.
        dataset: Dataset name.
        request: Request payload.
        target_file: Destination path.
        retries: Number of retries on failure.
        initial_delay: Seconds to wait before first retry; doubles each time.
    """
    async with semaphore:
        delay = initial_delay
        for attempt in range(retries):
            try:
                # Run the blocking retrieve + download sequence in a thread so it
                # does not block the event loop.
                await asyncio.to_thread(
                    lambda: client.retrieve(dataset, request).download(str(target_file))
                )
                print(f"Successfully downloaded {target_file}")
                return
            except Exception as e:
                print(
                    f"Attempt {attempt + 1}/{retries}: Failed to download {target_file}. Error: {e}"
                )
                if attempt < retries - 1:
                    print(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                    delay *= 2  # Exponential back-off
        # If we exit the loop without returning, the download has failed.
        print(f"Failed to download {target_file} after {retries} attempts.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download ERA5 data.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/era5",
        help="Directory to save downloaded files.",
    )
    parser.add_argument(
        "--start_year", type=int, default=2000, help="Start year for data download."
    )
    parser.add_argument(
        "--end_year", type=int, default=2025, help="End year for data download."
    )
    args = parser.parse_args()
    main(args.output_dir, args.start_year, args.end_year)

