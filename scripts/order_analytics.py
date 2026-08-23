from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
df=pd.read_csv(ROOT/'orders_dataset.csv')
out=ROOT/'results'
out.mkdir(exist_ok=True)
summary={
 'rows':len(df), 'return_rate':float(df.returned.mean()),
 'missing_rating_rate':float(df.rating_given.isna().mean()),
 'category_return_rates':df.groupby('product_category').returned.mean().round(4).to_dict(),
 'payment_return_rates':df.groupby('payment_method').returned.mean().round(4).to_dict(),
}
pd.DataFrame({'metric':summary.keys(),'value':[str(v) for v in summary.values()]}).to_csv(out/'order_analytics.csv',index=False)
print(summary)
