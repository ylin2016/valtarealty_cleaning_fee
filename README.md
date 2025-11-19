# Data Folder

Place your input data files here.

Expected files:

- `Property_Cohost.xlsx`
  - Sheet 0: property details
  - Sheet "Cleaning": cleaning fees
- `bookings.csv` **or** `bookings.xlsx`
  - Bookings data with `Listing`, `checkout_date`, `AvgDailyRate`, `guests`, etc.

The training script `src/train_cleaning_fee_model.py` expects:

- `checkout_date` as a date column.
- 2024-01-01 to 2025-09-30 as the analysis window (can be changed in the script).
- Consistent `Listing` names across property and bookings data.