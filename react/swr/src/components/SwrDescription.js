import '../css/SwrDescription.css';

const SwrDescription = (props) => {
    return (        
        <div className="container">
            <div className="row">
                <div className="col-12">
                    <div className="description text-left" >
                    <span className="font-weight-bold">Strategy:</span> Allocate {props.stock_alloc_pct}/{props.bond_alloc_pct} stocks/bonds, each year spend {props.withdrawal_fixed_pct}% of starting portfolio +  {props.withdrawal_variable_pct}% of current portfolio, with a floor of {props.withdrawal_floor_pct}% of starting portfolio.<br />
                    <span className="font-weight-bold">Outcome:</span> Initial spending <span data-testid="start_spend">{props.start_spend}</span>%, mean spending <span data-testid="mean_spend">{props.mean_spend}</span>% of starting portfolio, worst case <span data-testid="worst_spend">{props.worst_spend}</span>%, <span data-testid="pct_exhausted">{props.pct_exhausted}</span>% of retirement cohorts exhausted after 30 years.
                    <br /> &nbsp; 
                    </div>
                </div>
            </div>
        </div>
        );
    }

export default SwrDescription;