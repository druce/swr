import React, { Component } from 'react';
import { Button} from 'reactstrap';


class SwrQuickBtns extends Component {
    constructor(props) {
        super(props);
        this.bengen1 = this.bengen1.bind(this);
        this.bengen2 = this.bengen2.bind(this);
        this.ce1 = this.ce1.bind(this);
        this.ce2 = this.ce2.bind(this);
    }

    bengen1() {
        this.props.mainObj.setState({
            stock_alloc_pct: 50.0,
            bond_alloc_pct: 50.0,
            withdrawal_fixed_pct: 4.0,
            withdrawal_variable_pct: 0.0,
            withdrawal_floor_pct: 4.0
        }, this.props.mainObj.do_recalc);
    }

    bengen2() {
        this.props.mainObj.setState({
            stock_alloc_pct: 50.0,
            bond_alloc_pct: 50.0,
            withdrawal_fixed_pct: -1.0,
            withdrawal_variable_pct: 5.0,
            withdrawal_floor_pct: 4.0
        }, this.props.mainObj.do_recalc);
    }

    ce1() {
        this.props.mainObj.setState({
            stock_alloc_pct: 73.0,
            bond_alloc_pct: 27.0,
            withdrawal_fixed_pct: 3.5,
            withdrawal_variable_pct: 1.1,
            withdrawal_floor_pct: 3.8
        }, this.props.mainObj.do_recalc);
    }

    ce2() {
        this.props.mainObj.setState({
            stock_alloc_pct: 89.0,
            bond_alloc_pct: 11.0,
            withdrawal_fixed_pct: 0.7,
            withdrawal_variable_pct: 5.8,
            withdrawal_floor_pct: 3.4
        }, this.props.mainObj.do_recalc);
    }

    render() {
        return(
            <div className="container">
                <div className="row ">
                    <div className="col-12 col-md-9">
                            <Button outline color="secondary" onClick={this.bengen1}>Bengen original 4%</Button>
                            &nbsp;
                            <Button outline color="secondary" onClick={this.bengen2}>Relaxed 4%/5%</Button>
                            &nbsp;
                            <Button outline color="secondary" onClick={this.ce1}>Max CE, high risk aversion</Button>
                            &nbsp;
                            <Button outline color="secondary" onClick={this.ce2}>Max CE, medium risk aversion</Button>
                    </div>
                </div>
            </div>
        );
    }
}

export default SwrQuickBtns;