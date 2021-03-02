import React, { Component } from 'react';
import { Form, FormGroup, Label, Col} from 'reactstrap';
import Slider from 'rc-slider';
import 'rc-slider/assets/index.css';
import '../css/SwrSliders.css';

class SwrSliders extends Component {

    onStocksChange = value => {
        // console.log(value);
        this.props.mainObj.setState({
            stock_alloc_pct: value,
            bond_alloc_pct: 100 - value,
        })
        this.props.mainObj.calc_profile_df();
        this.props.mainObj.recalc_cohort_data();
    }

    onBondsChange = value => {
        // console.log(value);
        this.props.mainObj.setState({
            stock_alloc_pct: 100 - value,
            bond_alloc_pct: value,
        })        
        this.props.mainObj.calc_profile_df();
        this.props.mainObj.recalc_cohort_data();
    }

    onFixedChange = value => {
        // console.log(value);
        this.props.mainObj.setState({
            withdrawal_fixed_pct: value
        });
        this.props.mainObj.calc_profile_df();
        this.props.mainObj.recalc_cohort_data();
    }

    onVariableChange = value => {
        // console.log(value);
        this.props.mainObj.setState({
            withdrawal_variable_pct: value
        });
        this.props.mainObj.calc_profile_df();
        this.props.mainObj.recalc_cohort_data();
    }

    onFloorChange = value => {
        // console.log(value);
        this.props.mainObj.setState({
            withdrawal_floor_pct: value
        });
        this.props.mainObj.calc_profile_df();
        this.props.mainObj.recalc_cohort_data();
    }

    // onAfterChange = value => {
    //     // console.log(value);
    // };

    render() {
        return(
            <div className="col-12 col-md-9">
                <div className="col-12 col-md-8">
                    <h3 className="text-left">Allocation</h3>
                </div>
                <Form onSubmit={this.props.mainObj.handleSubmit}>
                <FormGroup row>
                            <Label htmlFor="stock_alloc_pct" md={3} className="text-right">Stocks %: {this.props.mainObj.state.stock_alloc_pct}</Label>
                            <Col md={9}>
                                <Slider 
                                    value={this.props.mainObj.state.stock_alloc_pct}
                                    min={0} 
                                    max={100} 
                                    step={1}
                                    onChange={this.onStocksChange}
                                />
                            </Col>
                    </FormGroup>
                    <FormGroup row>
                            <Label htmlFor="bond_alloc_pct" md={3} className="text-right">Bonds %: {this.props.mainObj.state.bond_alloc_pct}</Label>
                            <Col md={9}>
                                <Slider 
                                    value={this.props.mainObj.state.bond_alloc_pct}
                                    min={0} 
                                    max={100} 
                                    step={1}
                                    onChange={this.onBondsChange}
                                />
                            </Col>
                    </FormGroup>
                    
                    <div className="col-12 col-md-9">
                        <h3 className="text-left">Withdrawal</h3>
                    </div>

                    <FormGroup row>
                            <Label htmlFor="withdrawal_fixed" md={3} className="text-right">Fixed %: {this.props.mainObj.state.withdrawal_fixed_pct}</Label>
                            <Col md={9}>
                                <Slider 
                                    value={this.props.mainObj.state.withdrawal_fixed_pct}
                                    min={0} 
                                    max={5} 
                                    step={0.1}
                                    onChange={this.onFixedChange}
                                />
                            </Col>
                    </FormGroup>

                    <FormGroup row>
                            <Label htmlFor="withdrawal_variable" md={3} className="text-right">Variable %: {this.props.mainObj.state.withdrawal_variable_pct}</Label>
                            <Col md={9}>
                                <Slider 
                                    value={this.props.mainObj.state.withdrawal_variable_pct}
                                    min={0} 
                                    max={10} 
                                    step={0.1}
                                    onChange={this.onVariableChange}
                                />
                            </Col>
                    </FormGroup>

                    <FormGroup row>
                            <Label htmlFor="withdrawal_floor" md={3} className="text-right">Floor %: {this.props.mainObj.state.withdrawal_floor_pct}</Label>
                            <Col md={9}>
                                <Slider 
                                    value={this.props.mainObj.state.withdrawal_floor_pct}
                                    min={0} 
                                    max={5} 
                                    step={0.1}
                                    onChange={this.onFloorChange}
                                />
                            </Col>
                    </FormGroup>

                </Form>
            </div>
        );
    }
}

export default SwrSliders;