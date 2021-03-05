import '../css/SwrDescription.css';

const SwrDescription = (props) => {
    return (        
        <div className="container">
            <div className="row">
                <div className="col-12">
                    <div className="description text-left" >
                    <span className="font-weight-bold">Strategy:</span> Allocate {props.stock_alloc_pct}/{props.bond_alloc_pct} stocks/bonds, each year spend {props.withdrawal_fixed_pct}% of starting portfolio +  {props.withdrawal_variable_pct}% of current portfolio, with a floor of {props.withdrawal_floor_pct}% of starting portfolio.<br />
                    <span className="font-weight-bold">Outcome:</span> Initial spending {props.start_spend}%, mean spending {props.mean_spend}% of starting portfolio, worst case {props.worst_spend}%, {props.pct_exhausted}% of retirement cohorts exhausted after 30 years.
                    <br /> &nbsp; 
                    </div>
                </div>
            </div>
        </div>
        );
    }

export default SwrDescription;