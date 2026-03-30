use inception::primitive;
use typosaurus::collections::list::List as TList;
use typosaurus::collections::sp::Node;
use typosaurus::num::consts::*;

use crate::{
    And, ArrowGraph, ArrowNode, BoolToIndex2, Choose, ChunkVecOp, ConstOut, EnumerateVecOp, Eq,
    EqConstBool, Fanout, Fanout3, First, FlattenTripleResult, FromOp, GroupByIndexOp, GtConstUsize,
    Ident, Identified, Identity, InjectConst, JoinEither, LiftResult, MapErr, MapVec, Not,
    OptionThen, Or, Second, SequenceResult, SequenceResult3, SetInsertOp, Swap2, Switch, Switch3,
    Switch4, SwitchRef, Then, Third, TryFoldRange, TryFoldSliceMut, TryFoldVec, TryFromOp,
    TryIndexMutCtx, TryMapVec, VecLen, WrapOk, Zip, Zip3, ZipVecOp,
};

macro_rules! impl_identified {
    ($id:ty, [$($gen:ident),*], $ty:ty $(where $($where:tt)*)?) => {
        #[primitive(property = Ident)]
        impl<$($gen),*> Identified for $ty $(where $($where)*)? {
            type Id = $id;
        }
    };
}

macro_rules! impl_leaf_arrow_node {
    ([$($gen:ident),*], $ty:ty $(where $($where:tt)*)?) => {
        #[primitive(property = ArrowGraph)]
        impl<$($gen),*> ArrowNode for $ty $(where $($where)*)? {
            type Graph = Node<<Self as Identified>::Id, $ty>;
        }
    };
}

macro_rules! impl_leaf_primitive {
    ($id:ty, [$($gen:ident),*], $ty:ty $(where $($where:tt)*)?) => {
        impl_identified!($id, [$($gen),*], $ty $(where $($where)*)?);
        impl_leaf_arrow_node!([$($gen),*], $ty $(where $($where)*)?);
    };
}

impl_leaf_primitive!(U61, [T], Identity<T>);

impl_identified!(U62, [T, U], Choose<T, U>);
#[primitive(property = ArrowGraph)]
impl<T, U> ArrowNode for Choose<T, U>
where
    T: ArrowNode,
    U: ArrowNode,
{
    type Graph = typosaurus::parallel![<T as ArrowNode>::Graph, <U as ArrowNode>::Graph];
}

impl_leaf_primitive!(U63, [T], JoinEither<T>);

impl_identified!(U64, [Sel, B0, B1], Switch<Sel, B0, B1>);
#[primitive(property = ArrowGraph)]
impl<Sel, B0, B1> ArrowNode for Switch<Sel, B0, B1>
where
    Sel: ArrowNode,
    B0: ArrowNode,
    B1: ArrowNode,
{
    type Graph = TList<(
        <Sel as ArrowNode>::Graph,
        typosaurus::parallel![<B0 as ArrowNode>::Graph, <B1 as ArrowNode>::Graph],
    )>;
}

impl_identified!(U65, [In, B0, B1], SwitchRef<In, B0, B1>);
#[primitive(property = ArrowGraph)]
impl<In, B0, B1> ArrowNode for SwitchRef<In, B0, B1>
where
    B0: ArrowNode,
    B1: ArrowNode,
{
    type Graph = typosaurus::parallel![<B0 as ArrowNode>::Graph, <B1 as ArrowNode>::Graph];
}

impl_identified!(U66, [Sel, B0, B1, B2], Switch3<Sel, B0, B1, B2>);
#[primitive(property = ArrowGraph)]
impl<Sel, B0, B1, B2> ArrowNode for Switch3<Sel, B0, B1, B2>
where
    Sel: ArrowNode,
    B0: ArrowNode,
    B1: ArrowNode,
    B2: ArrowNode,
{
    type Graph = TList<(
        <Sel as ArrowNode>::Graph,
        typosaurus::parallel![
            <B0 as ArrowNode>::Graph,
            <B1 as ArrowNode>::Graph,
            <B2 as ArrowNode>::Graph
        ],
    )>;
}

impl_identified!(U67, [Sel, B0, B1, B2, B3], Switch4<Sel, B0, B1, B2, B3>);
#[primitive(property = ArrowGraph)]
impl<Sel, B0, B1, B2, B3> ArrowNode for Switch4<Sel, B0, B1, B2, B3>
where
    Sel: ArrowNode,
    B0: ArrowNode,
    B1: ArrowNode,
    B2: ArrowNode,
    B3: ArrowNode,
{
    type Graph = TList<(
        <Sel as ArrowNode>::Graph,
        typosaurus::parallel![
            <B0 as ArrowNode>::Graph,
            <B1 as ArrowNode>::Graph,
            <B2 as ArrowNode>::Graph,
            <B3 as ArrowNode>::Graph
        ],
    )>;
}

impl_identified!(U68, [Op1, Op2], Zip<Op1, Op2>);
#[primitive(property = ArrowGraph)]
impl<Op1, Op2> ArrowNode for Zip<Op1, Op2>
where
    Op1: ArrowNode,
    Op2: ArrowNode,
{
    type Graph = typosaurus::parallel![<Op1 as ArrowNode>::Graph, <Op2 as ArrowNode>::Graph];
}

impl_identified!(U69, [Head, Tail], Then<Head, Tail>);
#[primitive(property = ArrowGraph)]
impl<Head, Tail> ArrowNode for Then<Head, Tail>
where
    Head: ArrowNode,
    Tail: ArrowNode,
{
    type Graph = TList<(<Head as ArrowNode>::Graph, <Tail as ArrowNode>::Graph)>;
}

impl_leaf_primitive!(U70, [T], Fanout<T>);
impl_leaf_primitive!(U71, [A, B], First<A, B>);
impl_leaf_primitive!(U72, [A, B], Second<A, B>);
impl_leaf_primitive!(U73, [A, B], Swap2<A, B>);
impl_leaf_primitive!(U74, [], Not);
impl_leaf_primitive!(U75, [], And);
impl_leaf_primitive!(U76, [], Or);
impl_leaf_primitive!(U77, [], GtConstUsize);
impl_leaf_primitive!(U78, [], EqConstBool);
impl_leaf_primitive!(U79, [T], VecLen<T>);
impl_leaf_primitive!(U80, [T], Eq<T>);
impl_leaf_primitive!(U81, [A, B, C], Third<A, B, C>);
impl_leaf_primitive!(U82, [In, Out], FromOp<In, Out>);
impl_leaf_primitive!(U83, [In, Out, E], TryFromOp<In, Out, E>);
impl_leaf_primitive!(U84, [In, Out], ConstOut<In, Out> where Out: Clone);
impl_leaf_primitive!(U85, [], BoolToIndex2);
impl_leaf_primitive!(U86, [T, C], InjectConst<T, C> where C: Clone);

impl_identified!(U87, [Task, In, Out], LiftResult<Task, In, Out>);
#[primitive(property = ArrowGraph)]
impl<Task, In, Out> ArrowNode for LiftResult<Task, In, Out>
where
    Task: ArrowNode,
{
    type Graph = <Task as ArrowNode>::Graph;
}

impl_identified!(U88, [Task, In, C, E], OptionThen<Task, In, C, E>);
#[primitive(property = ArrowGraph)]
impl<Task, In, C, E> ArrowNode for OptionThen<Task, In, C, E>
where
    Task: ArrowNode,
{
    type Graph = <Task as ArrowNode>::Graph;
}

impl_leaf_primitive!(U89, [T, E], WrapOk<T, E>);

impl_identified!(U90, [Task, In, Ok, ErrIn, ErrOut], MapErr<Task, In, Ok, ErrIn, ErrOut>);
#[primitive(property = ArrowGraph)]
impl<Task, In, Ok, ErrIn, ErrOut> ArrowNode for MapErr<Task, In, Ok, ErrIn, ErrOut>
where
    Task: ArrowNode,
{
    type Graph = <Task as ArrowNode>::Graph;
}

impl_leaf_primitive!(U91, [A, B, E], SequenceResult<A, B, E>);
impl_leaf_primitive!(U92, [T], Fanout3<T>);

impl_identified!(U93, [Op1, Op2, Op3], Zip3<Op1, Op2, Op3>);
#[primitive(property = ArrowGraph)]
impl<Op1, Op2, Op3> ArrowNode for Zip3<Op1, Op2, Op3>
where
    Op1: ArrowNode,
    Op2: ArrowNode,
    Op3: ArrowNode,
{
    type Graph = typosaurus::parallel![
        <Op1 as ArrowNode>::Graph,
        <Op2 as ArrowNode>::Graph,
        <Op3 as ArrowNode>::Graph
    ];
}

impl_leaf_primitive!(U94, [A, B, C, E], SequenceResult3<A, B, C, E>);
impl_leaf_primitive!(U95, [A, B, C, E], FlattenTripleResult<A, B, C, E>);

impl_identified!(U96, [Step, State, Err], TryFoldRange<Step, State, Err>);
#[primitive(property = ArrowGraph)]
impl<Step, State, Err> ArrowNode for TryFoldRange<Step, State, Err>
where
    Step: ArrowNode,
{
    type Graph = <Step as ArrowNode>::Graph;
}

impl_leaf_primitive!(U97, [Ctx, Item, In, Out, Err], TryIndexMutCtx<Ctx, Item, In, Out, Err>);
impl_leaf_primitive!(U98, [Ctx, Item, State, Err], TryFoldSliceMut<Ctx, Item, State, Err>);

impl_identified!(U99, [Step, State, Item, Err], TryFoldVec<Step, State, Item, Err>);
#[primitive(property = ArrowGraph)]
impl<Step, State, Item, Err> ArrowNode for TryFoldVec<Step, State, Item, Err>
where
    Step: ArrowNode,
{
    type Graph = <Step as ArrowNode>::Graph;
}

impl_identified!(U100, [Step, InItem, OutItem], MapVec<Step, InItem, OutItem>);
#[primitive(property = ArrowGraph)]
impl<Step, InItem, OutItem> ArrowNode for MapVec<Step, InItem, OutItem>
where
    Step: ArrowNode,
{
    type Graph = <Step as ArrowNode>::Graph;
}

impl_identified!(U101, [Step, InItem, OutItem, Err], TryMapVec<Step, InItem, OutItem, Err>);
#[primitive(property = ArrowGraph)]
impl<Step, InItem, OutItem, Err> ArrowNode for TryMapVec<Step, InItem, OutItem, Err>
where
    Step: ArrowNode,
{
    type Graph = <Step as ArrowNode>::Graph;
}

impl_leaf_primitive!(U102, [T], SetInsertOp<T>);
impl_leaf_primitive!(U103, [T], EnumerateVecOp<T>);
impl_leaf_primitive!(U104, [A, B], ZipVecOp<A, B>);
impl_leaf_primitive!(U105, [T], ChunkVecOp<T>);
impl_leaf_primitive!(U106, [V], GroupByIndexOp<V>);
