import React, { Component } from 'react';
import { Form, FormGroup, Col, Label, Button} from 'reactstrap';
import Slider from 'rc-slider';
import 'rc-slider/assets/index.css';
import '../css/SwrSliders.css';

class SwrSliders extends Component {

    constructor(props) {
        super(props);
        this.bengen1 = this.bengen1.bind(this);
        this.ce1 = this.ce1.bind(this);
        this.ce2 = this.ce2.bind(this);
    }

    // slider callbacks
    onStocksChange = value => {
        // console.log(value);
        this.props.mainObj.setState({
            stock_alloc_pct: value,
            bond_alloc_pct: 100 - value,
        }, this.props.mainObj.do_recalc);
    }

    onBondsChange = value => {
        // console.log(value);
        this.props.mainObj.setState({
            stock_alloc_pct: 100 - value,
            bond_alloc_pct: value,
        }, this.props.mainObj.do_recalc);
    }

    onFixedChange = value => {
        // console.log(value);
        this.props.mainObj.setState({
            withdrawal_fixed_pct: value
        }, this.props.mainObj.do_recalc);
    }

    onVariableChange = value => {
        // console.log(value);
        this.props.mainObj.setState({
            withdrawal_variable_pct: value

        }, this.props.mainObj.do_recalc);
    }

    onFloorChange = value => {
        // console.log(value);
        this.props.mainObj.setState({
            withdrawal_floor_pct: value
        }, this.props.mainObj.do_recalc);
    }

    // quick button callbacks
    bengen1() {
        this.props.mainObj.setState({
            stock_alloc_pct: 50.0,
            bond_alloc_pct: 50.0,
            withdrawal_fixed_pct: 4.0,
            withdrawal_variable_pct: 0.0,
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
            withdrawal_fixed_pct: 2.6,
            withdrawal_variable_pct: 2.9,
            withdrawal_floor_pct: 3.5
        }, this.props.mainObj.do_recalc);
    }

    render() {
        return(
            <div className="container">
            <div className="row ">
    
            <div className="col-12 col-md-9">
                <div className="col-12 col-md-8">
                    <h3 className="text-left">Asset Allocation:</h3>
                </div>
                <Form onSubmit={this.props.mainObj.handleSubmit} classID="sliderForm">
                <FormGroup row className="align-items-center">
                            <Label data-testid="stock_alloc_pct_label" htmlFor="stock_alloc_pct" md={3} className="text-right">Stocks %: {this.props.mainObj.state.stock_alloc_pct}</Label>
                            <Col md={9}>
                                <Slider 
                                    value={this.props.mainObj.state.stock_alloc_pct}
                                    min={0} 
                                    max={100} 
                                    step={1}
                                    onChange={this.onStocksChange}
                                    onAfterChange={this.onStocksChange}
                                />
                            </Col>
                    </FormGroup>
                    <FormGroup row className="align-items-center">
                            <Label data-testid="bond_alloc_pct_label" htmlFor="bond_alloc_pct" md={3} className="text-right">Bonds %: {this.props.mainObj.state.bond_alloc_pct}</Label>
                            <Col md={9}>
                                <Slider 
                                    value={this.props.mainObj.state.bond_alloc_pct}
                                    min={0} 
                                    max={100} 
                                    step={1}
                                    onChange={this.onBondsChange}
                                    onAfterChange={this.onBondsChange}
                                />
                            </Col>
                    </FormGroup>
                    
                    <div className="col-12 col-md-9">
                        <h3 className="text-left">Annual Withdrawals:</h3>
                    </div>

                    <FormGroup row className="align-items-center">
                            <Label data-testid="withdrawal_fixed_label" htmlFor="withdrawal_fixed" md={3} className="text-right">Fixed %: {this.props.mainObj.state.withdrawal_fixed_pct}</Label>
                            <Col md={9}>
                                <Slider 
                                    value={this.props.mainObj.state.withdrawal_fixed_pct}
                                    min={-6} 
                                    max={6} 
                                    step={0.1}
                                    onChange={this.onFixedChange}
                                    onAfterChange={this.onFixedChange}
                                />
                            </Col>
                    </FormGroup>

                    <FormGroup row className="align-items-center">
                            <Label data-testid="withdrawal_variable_label" htmlFor="withdrawal_variable" md={3} className="text-right">Variable %: {this.props.mainObj.state.withdrawal_variable_pct}</Label>
                            <Col md={9}>
                                <Slider 
                                    value={this.props.mainObj.state.withdrawal_variable_pct}
                                    min={0} 
                                    max={10} 
                                    step={0.1}
                                    onChange={this.onVariableChange}
                                    onAfterChange={this.onVariableChange}
                                />
                            </Col>
                    </FormGroup>

                    <FormGroup row className="align-items-center">
                            <Label data-testid="withdrawal_floor_label" htmlFor="withdrawal_floor" md={3} className="text-right">Floor %: {this.props.mainObj.state.withdrawal_floor_pct}</Label>
                            <Col md={9}>
                                <Slider 
                                    value={this.props.mainObj.state.withdrawal_floor_pct}
                                    min={0} 
                                    max={6} 
                                    step={0.1}
                                    onChange={this.onFloorChange}
                                    onAfterChange={this.onFloorChange}
                                />
                            </Col>
                    </FormGroup>
                    <div className="container">
                    <div className="row ">
                        <div className="col-12">
                                <Button data-testid="button1" outline color="secondary" onClick={this.bengen1}>Bengen 4% rule</Button>
                                &nbsp;
                                <Button data-testid="button3" outline color="secondary" onClick={this.ce1}>Low risk rule (gamma=16)</Button>
                                &nbsp;
                                <Button  data-testid="button4" outline color="secondary" onClick={this.ce2}>High risk rule (gamma=4)</Button>
                        </div>
                    </div>
            </div>

                    <div className="col-12 col-md-9">
                        &nbsp; <br />
                        <h3 className="text-left">Historical Outcomes:</h3>
                    </div>


                </Form>
            </div>
            </div>
            </div>
        );
    }
}

export default SwrSliders;