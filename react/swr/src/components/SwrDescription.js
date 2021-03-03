import '../css/SwrDescription.css';

const SwrDescription = (props) => {
    return (        
        <div className="container">
            <div className="row">
                <div className="col-12">
                    <div className="description" {...props}>
                        <span className="font-weight-bold">Strategy:</span> Allocate {props.stock_alloc_pct}/{props.bond_alloc_pct} stocks/bonds, each year spend {props.withdrawal_fixed_pct}% of starting portfolio +  {props.withdrawal_variable_pct}% of current portfolio, with a floor of {props.withdrawal_floor_pct}% of starting portfolio<br /> &nbsp; 
                    </div>
                </div>
            </div>
        </div>
        );
    }

export default SwrDescription;