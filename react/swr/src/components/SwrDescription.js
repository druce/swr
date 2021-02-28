import '../css/SwrDescription.css';

const SwrDescription = (props) => {
    return <div className="description" {...props}>Strategy: {props.stock_alloc_pct}/{props.bond_alloc_pct} stocks/bonds, spend {props.withdrawal_fixed_pct}% of starting portfolio +  {props.withdrawal_variable_pct}% of current portfolio, with a floor of {props.withdrawal_floor_pct}% <br /> &nbsp; </div>
    }

export default SwrDescription;