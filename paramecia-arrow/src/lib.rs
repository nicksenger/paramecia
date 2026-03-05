use std::collections::HashSet;
use std::hash::Hash;
use std::marker::PhantomData;

use inception::*;

pub mod node_trace;
pub mod vis;
use vis::{Graph, Vis, Visualize};

use crate::vis::pretty_type;

#[inline]
pub fn forward_traced<Ctx, C>(
    combinator: &mut C,
    ctx: &mut Ctx,
    input: <C as Combinator<Ctx>>::In,
) -> <C as Combinator<Ctx>>::Out
where
    C: Combinator<Ctx>,
{
    crate::node_trace::trace_forward::<C, _>(|| combinator.forward(ctx, input))
}

pub trait CombinatorTraceExt<Ctx>: Combinator<Ctx> {
    #[inline]
    fn traced_forward(&mut self, ctx: &mut Ctx, input: Self::In) -> Self::Out
    where
        Self: Sized,
    {
        forward_traced(self, ctx, input)
    }
}

impl<Ctx, C> CombinatorTraceExt<Ctx> for C where C: Combinator<Ctx> {}

use typosaurus::collections::graph::{self, Combine, Empty};
use typosaurus::collections::set::{Set, Union};
use typosaurus::num::consts::{U0, U1, U2, U3, U42};
use typosaurus::{graph, set};
pub struct Node<T, U>(PhantomData<T>, PhantomData<U>);
#[inception(property = ArrowGraph, types)]
pub trait ArrowNode {
    #[induce(
        base = typosaurus::collections::set::Empty,
        merge = <(<Head as ArrowNode>::Id, <Tail as ArrowNode>::Id) as Union>::Out where { (<Head as ArrowNode>::Id, <Tail as ArrowNode>::Id): Union },
        merge_variant = <(<Head as ArrowNode>::Id, <Tail as ArrowNode>::Id) as Union>::Out where { (<Head as ArrowNode>::Id, <Tail as ArrowNode>::Id): Union },
        join = set![U0]
    )]
    type Id;

    #[induce(
        base = Empty,
        merge = Combine<<Head as ArrowNode>::Graph, Tail> where { (<Head as ArrowNode>::Graph, Tail): graph::Merge },
        merge_variant = Combine<<Head as ArrowNode>::Graph, Tail> where { (<Head as ArrowNode>::Graph, Tail): graph::Merge },
        join = <Fields as ArrowNode>::Graph
    )]
    type Graph;
}

#[cfg(test)]
mod graphtest {
    use typosaurus::collections::{
        graph::{Topo, Topological},
        list::Len,
    };

    use super::*;

    pub struct Foo;
    type FooId = set![U1];
    #[primitive(property = ArrowGraph)]
    impl ArrowNode for Foo {
        type Id = FooId;
        type Graph = graph! {(FooId, ()): []};
    }
    pub struct Bar;
    type BarId = set![U2];
    #[primitive(property = ArrowGraph)]
    impl ArrowNode for Bar {
        type Id = BarId;
        type Graph = graph! {(BarId, ()): []};
    }
    pub struct Baz;
    type BazId = set![U3];
    #[primitive(property = ArrowGraph)]
    impl ArrowNode for Baz {
        type Id = BazId;
        type Graph = graph! {(BazId, ()): []};
    }

    #[derive(Inception)]
    #[inception(properties = [ArrowGraph])]
    pub struct Waldo(Foo, Bar, Foo);

    #[derive(Inception)]
    #[inception(properties = [ArrowGraph])]
    pub struct Corge(Waldo, Waldo, Baz);

    #[test]
    fn test_arrowgraph() {
        type X = <Waldo as ArrowNode>::Graph;
        println!("{}", std::any::type_name::<X>());
    }
}

#[inception(property = Arrow, signature(input = In, output = Out))]
pub trait Combinator<Ctx = ()> {
    type In;
    type Out;

    fn forward(&mut self, ctx: &mut Ctx, input: Self::In) -> Self::Out;

    fn nothing(input: Self::In) -> Self::In {
        input
    }
    fn merge<H, R>(l: H, r: R, ctx: &mut Ctx, input: Self::In) -> <R as Combinator<Ctx>>::Out
    where
        H: Combinator<Ctx, In = Self::In>,
        R: Combinator<Ctx, In = <H as Combinator<Ctx>>::Out>,
    {
        let next = forward_traced(l.access(), ctx, input);
        forward_traced(r, ctx, next)
    }

    fn merge_variant_field<H, R>(_l: H, _r: R, ctx: &mut Ctx, input: Self::In) -> Self::In {
        let _ = (_l, _r, ctx);
        let _ = core::marker::PhantomData::<(H, R)>;
        input
    }

    fn join<F>(fields: F, ctx: &mut Ctx, input: Self::In) -> <F as Combinator<Ctx>>::Out
    where
        F: Combinator<Ctx, In = Self::In>,
    {
        forward_traced(fields, ctx, input)
    }
}

pub struct Identity<T>(PhantomData<T>);
#[primitive(property = Arrow)]
impl<Ctx, T> Combinator<Ctx> for Identity<T> {
    type In = T;
    type Out = T;
    fn forward(&mut self, _ctx: &mut Ctx, input: Self::In) -> Self::Out {
        input
    }
}
#[primitive(property = Visualize)]
impl<T> Vis for Identity<T> {
    fn visualize() -> Graph {
        Graph::leaf_with_edges("Identity", Some(&pretty_type(std::any::type_name::<T>())))
    }
}
impl<T> Default for Identity<T> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

pub struct WithUnitCtx<T>(pub T);
impl<Ctx, T> Combinator<Ctx> for WithUnitCtx<T>
where
    T: Combinator<()>,
{
    type In = <T as Combinator<()>>::In;
    type Out = <T as Combinator<()>>::Out;

    fn forward(&mut self, _ctx: &mut Ctx, input: Self::In) -> Self::Out {
        self.0.traced_forward(&mut (), input)
    }
}
#[primitive(property = Visualize)]
impl<T: Vis> Vis for WithUnitCtx<T> {
    fn visualize() -> Graph {
        <T as Vis>::visualize()
    }
}
impl<T> WithUnitCtx<T> {
    pub fn new(inner: T) -> Self {
        Self(inner)
    }
}

pub enum Either<Left, Right> {
    Left(Left),
    Right(Right),
}

pub trait SwitchIndex {
    fn index(&self) -> usize;
}

pub struct Choose<T, U>(bool, T, U);
#[primitive(property = Arrow)]
impl<Ctx, T, U> Combinator<Ctx> for Choose<T, U>
where
    T: Combinator<Ctx>,
    U: Combinator<Ctx, In = <T as Combinator<Ctx>>::In>,
{
    type In = <T as Combinator<Ctx>>::In;
    type Out = Either<<T as Combinator<Ctx>>::Out, <U as Combinator<Ctx>>::Out>;
    fn forward(&mut self, ctx: &mut Ctx, input: Self::In) -> Self::Out {
        if self.0 {
            Either::Left(self.1.traced_forward(ctx, input))
        } else {
            Either::Right(self.2.traced_forward(ctx, input))
        }
    }
}
#[primitive(property = Visualize)]
impl<T: Vis, U: Vis> Vis for Choose<T, U> {
    fn visualize() -> Graph {
        Graph::parallel(vec![<T as Vis>::visualize(), <U as Vis>::visualize()])
    }
}
impl<T, U> Choose<T, U> {
    pub fn new(enabled: bool, on_true: T, on_false: U) -> Self {
        Self(enabled, on_true, on_false)
    }
}

pub struct JoinEither<T>(PhantomData<T>);
#[primitive(property = Arrow)]
impl<Ctx, T> Combinator<Ctx> for JoinEither<T> {
    type In = Either<T, T>;
    type Out = T;
    fn forward(&mut self, _ctx: &mut Ctx, input: Self::In) -> Self::Out {
        match input {
            Either::Left(a) => a,
            Either::Right(b) => b,
        }
    }
}
#[primitive(property = Visualize)]
impl<T> Vis for JoinEither<T> {
    fn visualize() -> Graph {
        Graph::leaf_with_edges("JoinEither", Some(&pretty_type(std::any::type_name::<T>())))
    }
}

#[derive(Inception)]
#[inception(properties = [Arrow, Visualize])]
pub struct If<Left: 'static, Right: 'static, Out: 'static>(Choose<Left, Right>, JoinEither<Out>);
impl<Left, Right, Out> If<Left, Right, Out> {
    pub fn new(enabled: bool, on_true: Left, on_false: Right) -> Self {
        Self(
            Choose::new(enabled, on_true, on_false),
            JoinEither(PhantomData),
        )
    }
}

pub struct Switch<Sel, B0, B1>(Sel, B0, B1);
#[primitive(property = Arrow)]
impl<Ctx, Sel, B0, B1> Combinator<Ctx> for Switch<Sel, B0, B1>
where
    Sel: Combinator<Ctx>,
    <Sel as Combinator<Ctx>>::Out: SwitchIndex,
    <Sel as Combinator<Ctx>>::In: Clone,
    B0: Combinator<Ctx, In = <Sel as Combinator<Ctx>>::In>,
    B1: Combinator<Ctx, In = <Sel as Combinator<Ctx>>::In, Out = <B0 as Combinator<Ctx>>::Out>,
{
    type In = <Sel as Combinator<Ctx>>::In;
    type Out = <B0 as Combinator<Ctx>>::Out;

    fn forward(&mut self, ctx: &mut Ctx, input: Self::In) -> Self::Out {
        let idx = self.0.traced_forward(ctx, input.clone()).index();
        match idx {
            0 => self.1.traced_forward(ctx, input),
            _ => self.2.traced_forward(ctx, input),
        }
    }
}
#[primitive(property = Visualize)]
impl<Sel: Vis, B0: Vis, B1: Vis> Vis for Switch<Sel, B0, B1> {
    fn visualize() -> Graph {
        Graph::sequence(
            <Sel as Vis>::visualize(),
            Graph::parallel(vec![<B0 as Vis>::visualize(), <B1 as Vis>::visualize()]),
        )
    }
}
impl<Sel, B0, B1> Switch<Sel, B0, B1> {
    pub fn new(selector: Sel, b0: B0, b1: B1) -> Self {
        Self(selector, b0, b1)
    }
}

pub struct SwitchRef<In, B0, B1> {
    selector: fn(&In) -> usize,
    b0: B0,
    b1: B1,
}
#[primitive(property = Arrow)]
impl<Ctx, In, B0, B1> Combinator<Ctx> for SwitchRef<In, B0, B1>
where
    B0: Combinator<Ctx, In = In>,
    B1: Combinator<Ctx, In = In, Out = <B0 as Combinator<Ctx>>::Out>,
{
    type In = In;
    type Out = <B0 as Combinator<Ctx>>::Out;

    fn forward(&mut self, ctx: &mut Ctx, input: Self::In) -> Self::Out {
        match (self.selector)(&input) {
            0 => self.b0.traced_forward(ctx, input),
            _ => self.b1.traced_forward(ctx, input),
        }
    }
}
#[primitive(property = Visualize)]
impl<In, B0: Vis, B1: Vis> Vis for SwitchRef<In, B0, B1> {
    fn visualize() -> Graph {
        Graph::parallel(vec![<B0 as Vis>::visualize(), <B1 as Vis>::visualize()])
    }
}
impl<In, B0, B1> SwitchRef<In, B0, B1> {
    pub fn new(selector: fn(&In) -> usize, b0: B0, b1: B1) -> Self {
        Self { selector, b0, b1 }
    }
}

pub struct Switch3<Sel, B0, B1, B2>(Sel, B0, B1, B2);
#[primitive(property = Arrow)]
impl<Ctx, Sel, B0, B1, B2> Combinator<Ctx> for Switch3<Sel, B0, B1, B2>
where
    Sel: Combinator<Ctx>,
    <Sel as Combinator<Ctx>>::Out: SwitchIndex,
    <Sel as Combinator<Ctx>>::In: Clone,
    B0: Combinator<Ctx, In = <Sel as Combinator<Ctx>>::In>,
    B1: Combinator<Ctx, In = <Sel as Combinator<Ctx>>::In, Out = <B0 as Combinator<Ctx>>::Out>,
    B2: Combinator<Ctx, In = <Sel as Combinator<Ctx>>::In, Out = <B0 as Combinator<Ctx>>::Out>,
{
    type In = <Sel as Combinator<Ctx>>::In;
    type Out = <B0 as Combinator<Ctx>>::Out;

    fn forward(&mut self, ctx: &mut Ctx, input: Self::In) -> Self::Out {
        let idx = self.0.traced_forward(ctx, input.clone()).index();
        match idx {
            0 => self.1.traced_forward(ctx, input),
            1 => self.2.traced_forward(ctx, input),
            _ => self.3.traced_forward(ctx, input),
        }
    }
}
#[primitive(property = Visualize)]
impl<Sel: Vis, B0: Vis, B1: Vis, B2: Vis> Vis for Switch3<Sel, B0, B1, B2> {
    fn visualize() -> Graph {
        Graph::sequence(
            <Sel as Vis>::visualize(),
            Graph::parallel(vec![
                <B0 as Vis>::visualize(),
                <B1 as Vis>::visualize(),
                <B2 as Vis>::visualize(),
            ]),
        )
    }
}
impl<Sel, B0, B1, B2> Switch3<Sel, B0, B1, B2> {
    pub fn new(selector: Sel, b0: B0, b1: B1, b2: B2) -> Self {
        Self(selector, b0, b1, b2)
    }
}

pub struct Switch4<Sel, B0, B1, B2, B3>(Sel, B0, B1, B2, B3);
#[primitive(property = Arrow)]
impl<Ctx, Sel, B0, B1, B2, B3> Combinator<Ctx> for Switch4<Sel, B0, B1, B2, B3>
where
    Sel: Combinator<Ctx>,
    <Sel as Combinator<Ctx>>::Out: SwitchIndex,
    <Sel as Combinator<Ctx>>::In: Clone,
    B0: Combinator<Ctx, In = <Sel as Combinator<Ctx>>::In>,
    B1: Combinator<Ctx, In = <Sel as Combinator<Ctx>>::In, Out = <B0 as Combinator<Ctx>>::Out>,
    B2: Combinator<Ctx, In = <Sel as Combinator<Ctx>>::In, Out = <B0 as Combinator<Ctx>>::Out>,
    B3: Combinator<Ctx, In = <Sel as Combinator<Ctx>>::In, Out = <B0 as Combinator<Ctx>>::Out>,
{
    type In = <Sel as Combinator<Ctx>>::In;
    type Out = <B0 as Combinator<Ctx>>::Out;

    fn forward(&mut self, ctx: &mut Ctx, input: Self::In) -> Self::Out {
        let idx = self.0.traced_forward(ctx, input.clone()).index();
        match idx {
            0 => self.1.traced_forward(ctx, input),
            1 => self.2.traced_forward(ctx, input),
            2 => self.3.traced_forward(ctx, input),
            _ => self.4.traced_forward(ctx, input),
        }
    }
}
#[primitive(property = Visualize)]
impl<Sel: Vis, B0: Vis, B1: Vis, B2: Vis, B3: Vis> Vis for Switch4<Sel, B0, B1, B2, B3> {
    fn visualize() -> Graph {
        Graph::sequence(
            <Sel as Vis>::visualize(),
            Graph::parallel(vec![
                <B0 as Vis>::visualize(),
                <B1 as Vis>::visualize(),
                <B2 as Vis>::visualize(),
                <B3 as Vis>::visualize(),
            ]),
        )
    }
}
impl<Sel, B0, B1, B2, B3> Switch4<Sel, B0, B1, B2, B3> {
    pub fn new(selector: Sel, b0: B0, b1: B1, b2: B2, b3: B3) -> Self {
        Self(selector, b0, b1, b2, b3)
    }
}

pub struct Zip<Op1, Op2>(Op1, Op2);
#[primitive(property = Arrow)]
impl<Ctx, Op1, Op2> Combinator<Ctx> for Zip<Op1, Op2>
where
    Op1: Combinator<Ctx>,
    Op2: Combinator<Ctx>,
{
    type In = (<Op1 as Combinator<Ctx>>::In, <Op2 as Combinator<Ctx>>::In);
    type Out = (<Op1 as Combinator<Ctx>>::Out, <Op2 as Combinator<Ctx>>::Out);
    fn forward(&mut self, ctx: &mut Ctx, input: Self::In) -> Self::Out {
        (
            self.0.traced_forward(ctx, input.0),
            self.1.traced_forward(ctx, input.1),
        )
    }
}
#[primitive(property = Visualize)]
impl<Op1: Vis, Op2: Vis> Vis for Zip<Op1, Op2> {
    fn visualize() -> Graph {
        Graph::parallel(vec![<Op1 as Vis>::visualize(), <Op2 as Vis>::visualize()])
    }
}

impl<Op1, Op2> Zip<Op1, Op2> {
    pub fn new(left: Op1, right: Op2) -> Self {
        Self(left, right)
    }
}

pub struct Then<Head, Tail>(Head, Tail);
#[primitive(property = Arrow)]
impl<Ctx, Head, Tail> Combinator<Ctx> for Then<Head, Tail>
where
    Head: Combinator<Ctx>,
    Tail: Combinator<Ctx, In = <Head as Combinator<Ctx>>::Out>,
{
    type In = <Head as Combinator<Ctx>>::In;
    type Out = <Tail as Combinator<Ctx>>::Out;

    fn forward(&mut self, ctx: &mut Ctx, input: Self::In) -> Self::Out {
        let next = self.0.traced_forward(ctx, input);
        self.1.traced_forward(ctx, next)
    }
}
#[primitive(property = Visualize)]
impl<Head: Vis, Tail: Vis> Vis for Then<Head, Tail> {
    fn visualize() -> Graph {
        Graph::sequence(<Head as Vis>::visualize(), <Tail as Vis>::visualize())
    }
}
impl<Head, Tail> Then<Head, Tail> {
    pub fn new(head: Head, tail: Tail) -> Self {
        Self(head, tail)
    }
}

pub struct Fanout<T>(PhantomData<T>);
#[primitive(property = Arrow)]
impl<Ctx, T> Combinator<Ctx> for Fanout<T>
where
    T: Clone,
{
    type In = T;
    type Out = (T, T);
    fn forward(&mut self, _ctx: &mut Ctx, input: Self::In) -> Self::Out {
        (input.clone(), input)
    }
}
#[primitive(property = Visualize)]
impl<T> Vis for Fanout<T> {
    fn visualize() -> Graph {
        Graph::leaf_with_edges("Fanout", Some(&pretty_type(std::any::type_name::<T>())))
    }
}
impl<T> Default for Fanout<T> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

/// Selects the first element from a tuple.
pub struct First<A, B>(PhantomData<(A, B)>);
#[primitive(property = Arrow)]
impl<Ctx, A, B> Combinator<Ctx> for First<A, B> {
    type In = (A, B);
    type Out = A;

    fn forward(&mut self, _ctx: &mut Ctx, input: Self::In) -> Self::Out {
        let (left, _right) = input;
        left
    }
}
#[primitive(property = Visualize)]
impl<A, B> Vis for First<A, B> {
    fn visualize() -> Graph {
        Graph::leaf_with_edges("First", Some(&pretty_type(std::any::type_name::<A>())))
    }
}
impl<A, B> Default for First<A, B> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

/// Selects the second element from a tuple.
pub struct Second<A, B>(PhantomData<(A, B)>);
#[primitive(property = Arrow)]
impl<Ctx, A, B> Combinator<Ctx> for Second<A, B> {
    type In = (A, B);
    type Out = B;

    fn forward(&mut self, _ctx: &mut Ctx, input: Self::In) -> Self::Out {
        let (_left, right) = input;
        right
    }
}
#[primitive(property = Visualize)]
impl<A, B> Vis for Second<A, B> {
    fn visualize() -> Graph {
        Graph::leaf_with_edges("Second", Some(&pretty_type(std::any::type_name::<B>())))
    }
}
impl<A, B> Default for Second<A, B> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

/// Swaps tuple order `(A, B) -> (B, A)`.
pub struct Swap2<A, B>(PhantomData<(A, B)>);
#[primitive(property = Arrow)]
impl<Ctx, A, B> Combinator<Ctx> for Swap2<A, B> {
    type In = (A, B);
    type Out = (B, A);

    fn forward(&mut self, _ctx: &mut Ctx, input: Self::In) -> Self::Out {
        let (a, b) = input;
        (b, a)
    }
}
#[primitive(property = Visualize)]
impl<A, B> Vis for Swap2<A, B> {
    fn visualize() -> Graph {
        Graph::leaf_with_edges("Swap2", Some(&pretty_type(std::any::type_name::<(B, A)>())))
    }
}
impl<A, B> Default for Swap2<A, B> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

/// Boolean negation.
#[derive(Debug, Clone, Copy, Default)]
pub struct Not;
#[primitive(property = Arrow)]
impl<Ctx> Combinator<Ctx> for Not {
    type In = bool;
    type Out = bool;

    fn forward(&mut self, _ctx: &mut Ctx, input: Self::In) -> Self::Out {
        !input
    }
}
#[primitive(property = Visualize)]
impl Vis for Not {
    fn visualize() -> Graph {
        Graph::leaf_with_edges("Not", Some(&pretty_type(std::any::type_name::<bool>())))
    }
}

/// Boolean conjunction over tuple input `(lhs, rhs)`.
#[derive(Debug, Clone, Copy, Default)]
pub struct And;
#[primitive(property = Arrow)]
impl<Ctx> Combinator<Ctx> for And {
    type In = (bool, bool);
    type Out = bool;

    fn forward(&mut self, _ctx: &mut Ctx, input: Self::In) -> Self::Out {
        input.0 && input.1
    }
}
#[primitive(property = Visualize)]
impl Vis for And {
    fn visualize() -> Graph {
        Graph::leaf_with_edges("And", Some(&pretty_type(std::any::type_name::<bool>())))
    }
}

/// Boolean disjunction over tuple input `(lhs, rhs)`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Or;
#[primitive(property = Arrow)]
impl<Ctx> Combinator<Ctx> for Or {
    type In = (bool, bool);
    type Out = bool;

    fn forward(&mut self, _ctx: &mut Ctx, input: Self::In) -> Self::Out {
        input.0 || input.1
    }
}
#[primitive(property = Visualize)]
impl Vis for Or {
    fn visualize() -> Graph {
        Graph::leaf_with_edges("Or", Some(&pretty_type(std::any::type_name::<bool>())))
    }
}

/// Compares `input > threshold`.
#[derive(Debug, Clone, Copy)]
pub struct GtConstUsize(pub usize);
#[primitive(property = Arrow)]
impl<Ctx> Combinator<Ctx> for GtConstUsize {
    type In = usize;
    type Out = bool;

    fn forward(&mut self, _ctx: &mut Ctx, input: Self::In) -> Self::Out {
        input > self.0
    }
}
#[primitive(property = Visualize)]
impl Vis for GtConstUsize {
    fn visualize() -> Graph {
        Graph::leaf_with_edges(
            "GtConstUsize",
            Some(&pretty_type(std::any::type_name::<bool>())),
        )
    }
}

/// Compares `input == value`.
#[derive(Debug, Clone, Copy)]
pub struct EqConstBool(pub bool);
#[primitive(property = Arrow)]
impl<Ctx> Combinator<Ctx> for EqConstBool {
    type In = bool;
    type Out = bool;

    fn forward(&mut self, _ctx: &mut Ctx, input: Self::In) -> Self::Out {
        input == self.0
    }
}
#[primitive(property = Visualize)]
impl Vis for EqConstBool {
    fn visualize() -> Graph {
        Graph::leaf_with_edges(
            "EqConstBool",
            Some(&pretty_type(std::any::type_name::<bool>())),
        )
    }
}

/// Computes the length of a vector.
pub struct VecLen<T>(PhantomData<T>);
#[primitive(property = Arrow)]
impl<Ctx, T> Combinator<Ctx> for VecLen<T> {
    type In = Vec<T>;
    type Out = usize;

    fn forward(&mut self, _ctx: &mut Ctx, input: Self::In) -> Self::Out {
        input.len()
    }
}
#[primitive(property = Visualize)]
impl<T> Vis for VecLen<T> {
    fn visualize() -> Graph {
        Graph::leaf_with_edges("VecLen", Some(&pretty_type(std::any::type_name::<usize>())))
    }
}
impl<T> Default for VecLen<T> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

/// Equality comparison over tuple input `(lhs, rhs)`.
pub struct Eq<T>(PhantomData<T>);
#[primitive(property = Arrow)]
impl<Ctx, T> Combinator<Ctx> for Eq<T>
where
    T: PartialEq,
{
    type In = (T, T);
    type Out = bool;

    fn forward(&mut self, _ctx: &mut Ctx, input: Self::In) -> Self::Out {
        input.0 == input.1
    }
}
#[primitive(property = Visualize)]
impl<T> Vis for Eq<T> {
    fn visualize() -> Graph {
        Graph::leaf_with_edges("Eq", Some(&pretty_type(std::any::type_name::<bool>())))
    }
}
impl<T> Default for Eq<T> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

/// Selects the third element from a 3-tuple.
pub struct Third<A, B, C>(PhantomData<(A, B, C)>);
#[primitive(property = Arrow)]
impl<Ctx, A, B, C> Combinator<Ctx> for Third<A, B, C> {
    type In = (A, B, C);
    type Out = C;

    fn forward(&mut self, _ctx: &mut Ctx, input: Self::In) -> Self::Out {
        let (_a, _b, c) = input;
        c
    }
}
#[primitive(property = Visualize)]
impl<A, B, C> Vis for Third<A, B, C> {
    fn visualize() -> Graph {
        Graph::leaf_with_edges("Third", Some(&pretty_type(std::any::type_name::<C>())))
    }
}
impl<A, B, C> Default for Third<A, B, C> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

/// Converts `In` into `Out` via `From`.
pub struct FromOp<In, Out>(PhantomData<(In, Out)>);
#[primitive(property = Arrow)]
impl<Ctx, In, Out> Combinator<Ctx> for FromOp<In, Out>
where
    Out: From<In>,
{
    type In = In;
    type Out = Out;

    fn forward(&mut self, _ctx: &mut Ctx, input: Self::In) -> Self::Out {
        Out::from(input)
    }
}
#[primitive(property = Visualize)]
impl<In, Out> Vis for FromOp<In, Out> {
    fn visualize() -> Graph {
        Graph::leaf_with_edges("FromOp", Some(&pretty_type(std::any::type_name::<Out>())))
    }
}
impl<In, Out> Default for FromOp<In, Out> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

/// Fallible conversion `In -> Out` via `TryFrom`.
pub struct TryFromOp<In, Out, E>(PhantomData<(In, Out, E)>);
#[primitive(property = Arrow)]
impl<Ctx, In, Out, E> Combinator<Ctx> for TryFromOp<In, Out, E>
where
    Out: TryFrom<In, Error = E>,
{
    type In = In;
    type Out = Result<Out, E>;

    fn forward(&mut self, _ctx: &mut Ctx, input: Self::In) -> Self::Out {
        Out::try_from(input)
    }
}
#[primitive(property = Visualize)]
impl<In, Out, E> Vis for TryFromOp<In, Out, E> {
    fn visualize() -> Graph {
        Graph::leaf_with_edges(
            "TryFromOp",
            Some(&pretty_type(std::any::type_name::<Result<Out, E>>())),
        )
    }
}
impl<In, Out, E> Default for TryFromOp<In, Out, E> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

/// Emits a stored constant value, ignoring input.
pub struct ConstOut<In, Out: Clone>(Out, PhantomData<In>);
#[primitive(property = Arrow)]
impl<Ctx, In, Out: Clone> Combinator<Ctx> for ConstOut<In, Out> {
    type In = In;
    type Out = Out;

    fn forward(&mut self, _ctx: &mut Ctx, _input: Self::In) -> Self::Out {
        self.0.clone()
    }
}
#[primitive(property = Visualize)]
impl<In, Out: Clone> Vis for ConstOut<In, Out> {
    fn visualize() -> Graph {
        Graph::leaf_with_edges("ConstOut", Some(&pretty_type(std::any::type_name::<Out>())))
    }
}
impl<In, Out: Clone> ConstOut<In, Out> {
    pub fn new(out: Out) -> Self {
        Self(out, PhantomData)
    }
}

/// Switch index with two branches.
#[derive(Debug, Clone, Copy)]
pub struct Index2(pub usize);
impl SwitchIndex for Index2 {
    fn index(&self) -> usize {
        self.0
    }
}

/// Maps `true -> 0`, `false -> 1` for 2-way switching.
#[derive(Debug, Clone, Copy, Default)]
pub struct BoolToIndex2;
#[primitive(property = Arrow)]
impl<Ctx> Combinator<Ctx> for BoolToIndex2 {
    type In = bool;
    type Out = Index2;

    fn forward(&mut self, _ctx: &mut Ctx, input: Self::In) -> Self::Out {
        if input {
            Index2(0)
        } else {
            Index2(1)
        }
    }
}
#[primitive(property = Visualize)]
impl Vis for BoolToIndex2 {
    fn visualize() -> Graph {
        Graph::leaf_with_edges(
            "BoolToIndex2",
            Some(&pretty_type(std::any::type_name::<Index2>())),
        )
    }
}

/// Pairs the input with a stored constant, producing `(In, C)`.
///
/// Useful for injecting pre-computed values (weight tensors, configuration)
/// into a pipeline where they're needed as a second operand.
pub struct InjectConst<T, C: Clone>(C, PhantomData<T>);
#[primitive(property = Arrow)]
impl<Ctx, T, C: Clone> Combinator<Ctx> for InjectConst<T, C> {
    type In = T;
    type Out = (T, C);
    fn forward(&mut self, _ctx: &mut Ctx, input: Self::In) -> Self::Out {
        (input, self.0.clone())
    }
}
#[primitive(property = Visualize)]
impl<T, C: Clone> Vis for InjectConst<T, C> {
    fn visualize() -> Graph {
        Graph::leaf_with_edges(
            "InjectConst",
            Some(&pretty_type(std::any::type_name::<(T, C)>())),
        )
    }
}
impl<T, C: Clone> InjectConst<T, C> {
    pub fn new(constant: C) -> Self {
        Self(constant, PhantomData)
    }
}

pub trait ResultLike {
    type Ok;
    type Err;
    fn resolve(self) -> Result<Self::Ok, Self::Err>;
}
impl<A, B> ResultLike for Result<A, B> {
    type Ok = A;
    type Err = B;
    fn resolve(self) -> Result<A, B> {
        self
    }
}

pub struct LiftResult<Task, In, Out>(Task, PhantomData<In>, PhantomData<Out>);
#[primitive(property = Arrow)]
impl<Ctx, Task, In, Out> Combinator<Ctx> for LiftResult<Task, In, Out>
where
    In: ResultLike,
    Task: Combinator<Ctx, In = <In as ResultLike>::Ok, Out = Result<Out, <In as ResultLike>::Err>>,
{
    type In = In;
    type Out = Result<Out, <In as ResultLike>::Err>;
    fn forward(&mut self, ctx: &mut Ctx, input: Self::In) -> Self::Out {
        match input.resolve() {
            Ok(x) => self.0.traced_forward(ctx, x),
            Err(e) => Err(e),
        }
    }
}
#[primitive(property = Visualize)]
impl<Task: Vis, In, Out> Vis for LiftResult<Task, In, Out> {
    fn visualize() -> Graph {
        <Task as Vis>::visualize()
    }
}
impl<Task, In, Out> LiftResult<Task, In, Out> {
    pub fn new(t: Task) -> Self {
        Self(t, PhantomData, PhantomData)
    }
}

pub struct OptionThen<Task, In, C, E>(Task, PhantomData<(In, C, E)>);
#[primitive(property = Arrow)]
impl<Ctx, Task, In, C, E> Combinator<Ctx> for OptionThen<Task, In, C, E>
where
    Task: Combinator<Ctx, In = (In, C), Out = Result<In, E>>,
{
    type In = (In, Option<C>);
    type Out = Result<In, E>;

    fn forward(&mut self, ctx: &mut Ctx, input: Self::In) -> Self::Out {
        let (value, maybe) = input;
        match maybe {
            Some(conditional) => self.0.traced_forward(ctx, (value, conditional)),
            None => Ok(value),
        }
    }
}
#[primitive(property = Visualize)]
impl<Task: Vis, In, C, E> Vis for OptionThen<Task, In, C, E> {
    fn visualize() -> Graph {
        <Task as Vis>::visualize()
    }
}
impl<Task, In, C, E> OptionThen<Task, In, C, E> {
    pub fn new(task: Task) -> Self {
        Self(task, PhantomData)
    }
}

pub struct WrapOk<T, E>(PhantomData<(T, E)>);
#[primitive(property = Arrow)]
impl<Ctx, T, E> Combinator<Ctx> for WrapOk<T, E> {
    type In = T;
    type Out = Result<T, E>;

    fn forward(&mut self, _ctx: &mut Ctx, input: Self::In) -> Self::Out {
        Ok(input)
    }
}
#[primitive(property = Visualize)]
impl<T, E> Vis for WrapOk<T, E> {
    fn visualize() -> Graph {
        Graph::leaf_with_edges(
            "WrapOk",
            Some(&pretty_type(std::any::type_name::<Result<T, E>>())),
        )
    }
}
impl<T, E> Default for WrapOk<T, E> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

pub type MapOk<Task, In, Out, E> = LiftResult<Task, Result<In, E>, Out>;
pub type FanoutOk<T, E> = Fanout<Result<T, E>>;
pub type ZipOk<A, B, E> = SequenceResult<A, B, E>;

pub struct MapErr<Task, In, Ok, ErrIn, ErrOut>(Task, PhantomData<(In, Ok, ErrIn, ErrOut)>);
#[primitive(property = Arrow)]
impl<Ctx, Task, In, Ok, ErrIn, ErrOut> Combinator<Ctx> for MapErr<Task, In, Ok, ErrIn, ErrOut>
where
    Task: Combinator<Ctx, In = In, Out = Result<Ok, ErrIn>>,
    ErrOut: From<ErrIn>,
{
    type In = In;
    type Out = Result<Ok, ErrOut>;
    fn forward(&mut self, ctx: &mut Ctx, input: Self::In) -> Self::Out {
        self.0.traced_forward(ctx, input).map_err(ErrOut::from)
    }
}
#[primitive(property = Visualize)]
impl<Task: Vis, In, Ok, ErrIn, ErrOut> Vis for MapErr<Task, In, Ok, ErrIn, ErrOut> {
    fn visualize() -> Graph {
        <Task as Vis>::visualize()
    }
}
impl<Task, In, Ok, ErrIn, ErrOut> MapErr<Task, In, Ok, ErrIn, ErrOut> {
    pub fn new(task: Task) -> Self {
        Self(task, PhantomData)
    }
}
impl<Task, In, Ok, ErrIn, ErrOut> Default for MapErr<Task, In, Ok, ErrIn, ErrOut>
where
    Task: Default,
{
    fn default() -> Self {
        Self(Task::default(), PhantomData)
    }
}

pub struct SequenceResult<A, B, E>(PhantomData<(A, B, E)>);
#[primitive(property = Arrow)]
impl<Ctx, A, B, E> Combinator<Ctx> for SequenceResult<A, B, E> {
    type In = (Result<A, E>, Result<B, E>);
    type Out = Result<(A, B), E>;

    fn forward(&mut self, _ctx: &mut Ctx, (left, right): Self::In) -> Self::Out {
        Ok((left?, right?))
    }
}
#[primitive(property = Visualize)]
impl<A, B, E> Vis for SequenceResult<A, B, E> {
    fn visualize() -> Graph {
        Graph::leaf_with_edges(
            "SequenceResult",
            Some(&pretty_type(std::any::type_name::<Result<(A, B), E>>())),
        )
    }
}
impl<A, B, E> Default for SequenceResult<A, B, E> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

pub struct Fanout3<T>(PhantomData<T>);
#[primitive(property = Arrow)]
impl<Ctx, T> Combinator<Ctx> for Fanout3<T>
where
    T: Clone,
{
    type In = T;
    type Out = (T, T, T);
    fn forward(&mut self, _ctx: &mut Ctx, input: Self::In) -> Self::Out {
        let a = input.clone();
        let b = input.clone();
        (a, b, input)
    }
}
#[primitive(property = Visualize)]
impl<T> Vis for Fanout3<T> {
    fn visualize() -> Graph {
        Graph::leaf_with_edges(
            "Fanout3",
            Some(&pretty_type(std::any::type_name::<(T, T, T)>())),
        )
    }
}
impl<T> Default for Fanout3<T> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

pub struct Zip3<Op1, Op2, Op3>(Op1, Op2, Op3);
#[primitive(property = Arrow)]
impl<Ctx, Op1, Op2, Op3> Combinator<Ctx> for Zip3<Op1, Op2, Op3>
where
    Op1: Combinator<Ctx>,
    Op2: Combinator<Ctx>,
    Op3: Combinator<Ctx>,
{
    type In = (
        <Op1 as Combinator<Ctx>>::In,
        <Op2 as Combinator<Ctx>>::In,
        <Op3 as Combinator<Ctx>>::In,
    );
    type Out = (
        <Op1 as Combinator<Ctx>>::Out,
        <Op2 as Combinator<Ctx>>::Out,
        <Op3 as Combinator<Ctx>>::Out,
    );
    fn forward(&mut self, ctx: &mut Ctx, input: Self::In) -> Self::Out {
        (
            self.0.traced_forward(ctx, input.0),
            self.1.traced_forward(ctx, input.1),
            self.2.traced_forward(ctx, input.2),
        )
    }
}
#[primitive(property = Visualize)]
impl<Op1: Vis, Op2: Vis, Op3: Vis> Vis for Zip3<Op1, Op2, Op3> {
    fn visualize() -> Graph {
        Graph::parallel(vec![
            <Op1 as Vis>::visualize(),
            <Op2 as Vis>::visualize(),
            <Op3 as Vis>::visualize(),
        ])
    }
}
impl<Op1, Op2, Op3> Zip3<Op1, Op2, Op3> {
    pub fn new(a: Op1, b: Op2, c: Op3) -> Self {
        Self(a, b, c)
    }
}

pub struct SequenceResult3<A, B, C, E>(PhantomData<(A, B, C, E)>);
#[primitive(property = Arrow)]
impl<Ctx, A, B, T, E> Combinator<Ctx> for SequenceResult3<A, B, T, E> {
    type In = (Result<A, E>, Result<B, E>, Result<T, E>);
    type Out = Result<(A, B, T), E>;

    fn forward(&mut self, _ctx: &mut Ctx, (a, b, c): Self::In) -> Self::Out {
        Ok((a?, b?, c?))
    }
}
#[primitive(property = Visualize)]
impl<A, B, C, E> Vis for SequenceResult3<A, B, C, E> {
    fn visualize() -> Graph {
        Graph::leaf_with_edges(
            "SequenceResult3",
            Some(&pretty_type(std::any::type_name::<Result<(A, B, C), E>>())),
        )
    }
}
impl<A, B, C, E> Default for SequenceResult3<A, B, C, E> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

pub type ZipOk3<A, B, C, E> = SequenceResult3<A, B, C, E>;
pub type FanoutOk3<T, E> = Fanout3<Result<T, E>>;

pub struct FlattenTripleResult<A, B, C, E>(PhantomData<(A, B, C, E)>);
#[primitive(property = Arrow)]
impl<Ctx, A, B, C, E> Combinator<Ctx> for FlattenTripleResult<A, B, C, E> {
    type In = (A, (B, C));
    type Out = Result<(A, B, C), E>;

    fn forward(&mut self, _ctx: &mut Ctx, input: Self::In) -> Self::Out {
        let (a, (b, c)) = input;
        Ok((a, b, c))
    }
}
#[primitive(property = Visualize)]
impl<A, B, C, E> Vis for FlattenTripleResult<A, B, C, E> {
    fn visualize() -> Graph {
        Graph::leaf_with_edges(
            "FlattenTripleResult",
            Some(&pretty_type(std::any::type_name::<Result<(A, B, C), E>>())),
        )
    }
}
impl<A, B, C, E> Default for FlattenTripleResult<A, B, C, E> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

pub struct TryFoldRange<Step, State, Err>(Step, PhantomData<(State, Err)>);
#[primitive(property = Arrow)]
impl<Ctx, Step, State, Err> Combinator<Ctx> for TryFoldRange<Step, State, Err>
where
    Step: Combinator<Ctx, In = (State, usize), Out = Result<State, Err>>,
{
    type In = (State, std::ops::Range<usize>);
    type Out = Result<State, Err>;

    fn forward(&mut self, ctx: &mut Ctx, (mut state, range): Self::In) -> Self::Out {
        for idx in range {
            state = self.0.traced_forward(ctx, (state, idx))?;
        }
        Ok(state)
    }
}
#[primitive(property = Visualize)]
impl<Step: Vis, State, Err> Vis for TryFoldRange<Step, State, Err> {
    fn visualize() -> Graph {
        <Step as Vis>::visualize()
    }
}
impl<Step, State, Err> TryFoldRange<Step, State, Err> {
    pub fn new(step: Step) -> Self {
        Self(step, PhantomData)
    }
}
impl<Step, State, Err> Default for TryFoldRange<Step, State, Err>
where
    Step: Default,
{
    fn default() -> Self {
        Self(Step::default(), PhantomData)
    }
}

pub struct TryIndexMutCtx<Ctx, Item, In, Out, Err> {
    slice: for<'a> fn(&'a mut Ctx) -> &'a mut [Item],
    index: fn(&In) -> usize,
    step: fn(&mut Item, In) -> Result<Out, Err>,
    out_of_bounds: fn(usize, usize) -> Err,
}
#[primitive(property = Arrow)]
impl<Ctx, Item, In, Out, Err> Combinator<Ctx> for TryIndexMutCtx<Ctx, Item, In, Out, Err> {
    type In = In;
    type Out = Result<Out, Err>;

    fn forward(&mut self, ctx: &mut Ctx, input: Self::In) -> Self::Out {
        let idx = (self.index)(&input);
        let slice = (self.slice)(ctx);
        let len = slice.len();
        let item = slice
            .get_mut(idx)
            .ok_or_else(|| (self.out_of_bounds)(idx, len))?;
        (self.step)(item, input)
    }
}
#[primitive(property = Visualize)]
impl<Ctx, Item, In, Out, Err> Vis for TryIndexMutCtx<Ctx, Item, In, Out, Err> {
    fn visualize() -> Graph {
        Graph::leaf_with_edges(
            "TryIndexMutCtx",
            Some(&pretty_type(std::any::type_name::<Result<Out, Err>>())),
        )
    }
}
impl<Ctx, Item, In, Out, Err> TryIndexMutCtx<Ctx, Item, In, Out, Err> {
    pub fn new(
        slice: for<'a> fn(&'a mut Ctx) -> &'a mut [Item],
        index: fn(&In) -> usize,
        step: fn(&mut Item, In) -> Result<Out, Err>,
        out_of_bounds: fn(usize, usize) -> Err,
    ) -> Self {
        Self {
            slice,
            index,
            step,
            out_of_bounds,
        }
    }
}

pub struct TryFoldSliceMut<Ctx, Item, State, Err> {
    slice: for<'a> fn(&'a mut Ctx) -> &'a mut [Item],
    step: fn(&mut Item, State, usize) -> Result<State, Err>,
    out_of_bounds: fn(usize, usize) -> Err,
}
#[primitive(property = Arrow)]
impl<Ctx, Item, State, Err> Combinator<Ctx> for TryFoldSliceMut<Ctx, Item, State, Err> {
    type In = (State, std::ops::Range<usize>);
    type Out = Result<State, Err>;

    fn forward(&mut self, ctx: &mut Ctx, (mut state, range): Self::In) -> Self::Out {
        let slice = (self.slice)(ctx);
        let len = slice.len();
        for idx in range {
            let item = slice
                .get_mut(idx)
                .ok_or_else(|| (self.out_of_bounds)(idx, len))?;
            state = (self.step)(item, state, idx)?;
        }
        Ok(state)
    }
}
#[primitive(property = Visualize)]
impl<Ctx, Item, State, Err> Vis for TryFoldSliceMut<Ctx, Item, State, Err> {
    fn visualize() -> Graph {
        Graph::leaf_with_edges(
            "TryFoldSliceMut",
            Some(&pretty_type(std::any::type_name::<Result<State, Err>>())),
        )
    }
}
impl<Ctx, Item, State, Err> TryFoldSliceMut<Ctx, Item, State, Err> {
    pub fn new(
        slice: for<'a> fn(&'a mut Ctx) -> &'a mut [Item],
        step: fn(&mut Item, State, usize) -> Result<State, Err>,
        out_of_bounds: fn(usize, usize) -> Err,
    ) -> Self {
        Self {
            slice,
            step,
            out_of_bounds,
        }
    }
}

pub struct TryFoldVec<Step, State, Item, Err>(Step, PhantomData<(State, Item, Err)>);
#[primitive(property = Arrow)]
impl<Ctx, Step, State, Item, Err> Combinator<Ctx> for TryFoldVec<Step, State, Item, Err>
where
    Step: Combinator<Ctx, In = (State, Item), Out = Result<State, Err>>,
{
    type In = (State, Vec<Item>);
    type Out = Result<State, Err>;

    fn forward(&mut self, ctx: &mut Ctx, (mut state, items): Self::In) -> Self::Out {
        for item in items {
            state = self.0.traced_forward(ctx, (state, item))?;
        }
        Ok(state)
    }
}
#[primitive(property = Visualize)]
impl<Step: Vis, State, Item, Err> Vis for TryFoldVec<Step, State, Item, Err> {
    fn visualize() -> Graph {
        <Step as Vis>::visualize()
    }
}
impl<Step, State, Item, Err> TryFoldVec<Step, State, Item, Err> {
    pub fn new(step: Step) -> Self {
        Self(step, PhantomData)
    }
}
impl<Step, State, Item, Err> Default for TryFoldVec<Step, State, Item, Err>
where
    Step: Default,
{
    fn default() -> Self {
        Self(Step::default(), PhantomData)
    }
}

pub struct MapVec<Step, InItem, OutItem>(Step, PhantomData<(InItem, OutItem)>);
#[primitive(property = Arrow)]
impl<Ctx, Step, InItem, OutItem> Combinator<Ctx> for MapVec<Step, InItem, OutItem>
where
    Step: Combinator<Ctx, In = InItem, Out = OutItem>,
{
    type In = Vec<InItem>;
    type Out = Vec<OutItem>;

    fn forward(&mut self, ctx: &mut Ctx, items: Self::In) -> Self::Out {
        let mut out = Vec::with_capacity(items.len());
        for item in items {
            out.push(self.0.traced_forward(ctx, item));
        }
        out
    }
}
#[primitive(property = Visualize)]
impl<Step: Vis, InItem, OutItem> Vis for MapVec<Step, InItem, OutItem> {
    fn visualize() -> Graph {
        <Step as Vis>::visualize()
    }
}
impl<Step, InItem, OutItem> MapVec<Step, InItem, OutItem> {
    pub fn new(step: Step) -> Self {
        Self(step, PhantomData)
    }
}
impl<Step, InItem, OutItem> Default for MapVec<Step, InItem, OutItem>
where
    Step: Default,
{
    fn default() -> Self {
        Self(Step::default(), PhantomData)
    }
}

pub struct TryMapVec<Step, InItem, OutItem, Err>(Step, PhantomData<(InItem, OutItem, Err)>);
#[primitive(property = Arrow)]
impl<Ctx, Step, InItem, OutItem, Err> Combinator<Ctx> for TryMapVec<Step, InItem, OutItem, Err>
where
    Step: Combinator<Ctx, In = InItem, Out = Result<OutItem, Err>>,
{
    type In = Vec<InItem>;
    type Out = Result<Vec<OutItem>, Err>;

    fn forward(&mut self, ctx: &mut Ctx, items: Self::In) -> Self::Out {
        let mut out = Vec::with_capacity(items.len());
        for item in items {
            out.push(self.0.traced_forward(ctx, item)?);
        }
        Ok(out)
    }
}
#[primitive(property = Visualize)]
impl<Step: Vis, InItem, OutItem, Err> Vis for TryMapVec<Step, InItem, OutItem, Err> {
    fn visualize() -> Graph {
        <Step as Vis>::visualize()
    }
}
impl<Step, InItem, OutItem, Err> TryMapVec<Step, InItem, OutItem, Err> {
    pub fn new(step: Step) -> Self {
        Self(step, PhantomData)
    }
}
impl<Step, InItem, OutItem, Err> Default for TryMapVec<Step, InItem, OutItem, Err>
where
    Step: Default,
{
    fn default() -> Self {
        Self(Step::default(), PhantomData)
    }
}

pub struct SetInsertOp<T>(PhantomData<T>);
#[primitive(property = Arrow)]
impl<Ctx, T> Combinator<Ctx> for SetInsertOp<T>
where
    T: std::cmp::Eq + Hash,
{
    type In = (HashSet<T>, T);
    type Out = HashSet<T>;

    fn forward(&mut self, _ctx: &mut Ctx, input: Self::In) -> Self::Out {
        let (mut set, item) = input;
        set.insert(item);
        set
    }
}
#[primitive(property = Visualize)]
impl<T> Vis for SetInsertOp<T> {
    fn visualize() -> Graph {
        Graph::leaf_with_edges(
            "SetInsert",
            Some(&pretty_type(std::any::type_name::<HashSet<T>>())),
        )
    }
}
impl<T> Default for SetInsertOp<T> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum VecStructureError {
    #[error("zip length mismatch: left {left}, right {right}")]
    ZipLenMismatch { left: usize, right: usize },
    #[error("chunk size must be > 0")]
    ChunkSizeZero,
    #[error("chunk length mismatch: len {len} not divisible by chunk_size {chunk_size}")]
    ChunkLenMismatch { len: usize, chunk_size: usize },
    #[error("group index out of bounds: index {index}, num_groups {num_groups}")]
    GroupIndexOutOfBounds { index: usize, num_groups: usize },
}

pub struct EnumerateVecOp<T>(PhantomData<T>);
#[primitive(property = Arrow)]
impl<Ctx, T> Combinator<Ctx> for EnumerateVecOp<T> {
    type In = Vec<T>;
    type Out = Vec<(usize, T)>;

    fn forward(&mut self, _ctx: &mut Ctx, input: Self::In) -> Self::Out {
        input.into_iter().enumerate().collect()
    }
}
#[primitive(property = Visualize)]
impl<T> Vis for EnumerateVecOp<T> {
    fn visualize() -> Graph {
        Graph::leaf_with_edges(
            "EnumerateVec",
            Some(&pretty_type(std::any::type_name::<Vec<(usize, T)>>())),
        )
    }
}
impl<T> Default for EnumerateVecOp<T> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

pub struct ZipVecOp<A, B>(PhantomData<(A, B)>);
#[primitive(property = Arrow)]
impl<Ctx, A, B> Combinator<Ctx> for ZipVecOp<A, B> {
    type In = (Vec<A>, Vec<B>);
    type Out = Result<Vec<(A, B)>, VecStructureError>;

    fn forward(&mut self, _ctx: &mut Ctx, input: Self::In) -> Self::Out {
        let (left, right) = input;
        if left.len() != right.len() {
            return Err(VecStructureError::ZipLenMismatch {
                left: left.len(),
                right: right.len(),
            });
        }
        Ok(left.into_iter().zip(right).collect())
    }
}
#[primitive(property = Visualize)]
impl<A, B> Vis for ZipVecOp<A, B> {
    fn visualize() -> Graph {
        Graph::leaf_with_edges(
            "ZipVec",
            Some(&pretty_type(std::any::type_name::<
                Result<Vec<(A, B)>, VecStructureError>,
            >())),
        )
    }
}
impl<A, B> Default for ZipVecOp<A, B> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

pub struct ChunkVecOp<T> {
    chunk_size: usize,
    marker: PhantomData<T>,
}
#[primitive(property = Arrow)]
impl<Ctx, T> Combinator<Ctx> for ChunkVecOp<T> {
    type In = Vec<T>;
    type Out = Result<Vec<Vec<T>>, VecStructureError>;

    fn forward(&mut self, _ctx: &mut Ctx, input: Self::In) -> Self::Out {
        if self.chunk_size == 0 {
            return Err(VecStructureError::ChunkSizeZero);
        }
        if input.len() % self.chunk_size != 0 {
            return Err(VecStructureError::ChunkLenMismatch {
                len: input.len(),
                chunk_size: self.chunk_size,
            });
        }
        let mut chunks = Vec::with_capacity(input.len() / self.chunk_size);
        let mut iter = input.into_iter();
        loop {
            let mut chunk = Vec::with_capacity(self.chunk_size);
            for _ in 0..self.chunk_size {
                if let Some(item) = iter.next() {
                    chunk.push(item);
                } else {
                    break;
                }
            }
            if chunk.is_empty() {
                break;
            }
            chunks.push(chunk);
        }
        Ok(chunks)
    }
}
#[primitive(property = Visualize)]
impl<T> Vis for ChunkVecOp<T> {
    fn visualize() -> Graph {
        Graph::leaf_with_edges(
            "ChunkVec",
            Some(&pretty_type(std::any::type_name::<
                Result<Vec<Vec<T>>, VecStructureError>,
            >())),
        )
    }
}
impl<T> ChunkVecOp<T> {
    pub fn new(chunk_size: usize) -> Self {
        Self {
            chunk_size,
            marker: PhantomData,
        }
    }
}

pub struct GroupByIndexOp<V>(PhantomData<V>);
#[primitive(property = Arrow)]
impl<Ctx, V> Combinator<Ctx> for GroupByIndexOp<V> {
    type In = (usize, Vec<(usize, V)>);
    type Out = Result<Vec<Vec<V>>, VecStructureError>;

    fn forward(&mut self, _ctx: &mut Ctx, input: Self::In) -> Self::Out {
        let (num_groups, pairs) = input;
        let mut groups = (0..num_groups).map(|_| Vec::new()).collect::<Vec<_>>();
        for (index, value) in pairs {
            let Some(group) = groups.get_mut(index) else {
                return Err(VecStructureError::GroupIndexOutOfBounds { index, num_groups });
            };
            group.push(value);
        }
        Ok(groups)
    }
}
#[primitive(property = Visualize)]
impl<V> Vis for GroupByIndexOp<V> {
    fn visualize() -> Graph {
        Graph::leaf_with_edges(
            "GroupByIndex",
            Some(&pretty_type(std::any::type_name::<
                Result<Vec<Vec<V>>, VecStructureError>,
            >())),
        )
    }
}
impl<V> Default for GroupByIndexOp<V> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    pub struct RefToText<T>(PhantomData<T>);
    pub struct RefCharsArr<const N: usize>;
    pub struct Exclaim;
    pub struct AddIndex;
    pub struct AddItem;
    pub struct ToText;
    pub struct ToOddFail;
    pub struct FailI32;

    #[primitive(property = Arrow)]
    impl<T> Combinator for RefToText<T>
    where
        T: std::fmt::Display,
    {
        type In = T;
        type Out = String;

        fn forward(&mut self, _ctx: &mut (), input: Self::In) -> Self::Out {
            format!("{input}")
        }
    }
    impl<T> Default for RefToText<T> {
        fn default() -> Self {
            Self(PhantomData)
        }
    }

    #[primitive(property = Arrow)]
    impl<const N: usize> Combinator for RefCharsArr<N> {
        type In = String;
        type Out = Result<[char; N], Vec<char>>;

        fn forward(&mut self, _ctx: &mut (), input: Self::In) -> Self::Out {
            input.chars().collect::<Vec<_>>().try_into()
        }
    }

    #[primitive(property = Arrow)]
    impl Combinator for Exclaim {
        type In = [char; 2];
        type Out = Result<[char; 3], Vec<char>>;

        fn forward(&mut self, _ctx: &mut (), [a, b]: Self::In) -> Self::Out {
            Ok([a, b, '!'])
        }
    }

    #[primitive(property = Arrow)]
    impl Combinator for AddIndex {
        type In = (usize, usize);
        type Out = Result<usize, &'static str>;

        fn forward(&mut self, _ctx: &mut (), (acc, idx): Self::In) -> Self::Out {
            Ok(acc + idx)
        }
    }

    #[primitive(property = Arrow)]
    impl Combinator for AddItem {
        type In = (usize, usize);
        type Out = Result<usize, &'static str>;

        fn forward(&mut self, _ctx: &mut (), (acc, item): Self::In) -> Self::Out {
            Ok(acc + item)
        }
    }

    #[primitive(property = Arrow)]
    impl Combinator for ToText {
        type In = usize;
        type Out = String;

        fn forward(&mut self, _ctx: &mut (), input: Self::In) -> Self::Out {
            input.to_string()
        }
    }

    #[primitive(property = Arrow)]
    impl Combinator for ToOddFail {
        type In = usize;
        type Out = Result<usize, &'static str>;

        fn forward(&mut self, _ctx: &mut (), input: Self::In) -> Self::Out {
            if input % 2 == 0 {
                Ok(input)
            } else {
                Err("odd")
            }
        }
    }

    #[primitive(property = Arrow)]
    impl Combinator for FailI32 {
        type In = usize;
        type Out = Result<usize, i32>;

        fn forward(&mut self, _ctx: &mut (), _input: Self::In) -> Self::Out {
            Err(7)
        }
    }

    #[derive(Inception)]
    #[inception(properties = [Arrow])]
    struct TestArrow(RefToText<u32>, Fanout<String>, Zip<SubArrow, SubArrow>);

    type TryExclaim = LiftResult<Exclaim, Result<[char; 2], Vec<char>>, [char; 3]>;

    #[derive(Inception)]
    #[inception(properties = [Arrow])]
    struct SubArrow(RefCharsArr<2>, TryExclaim);

    fn sub_arrow() -> SubArrow {
        SubArrow(RefCharsArr, LiftResult::new(Exclaim))
    }

    #[test]
    fn test_arrow() {
        let mut arrow = TestArrow(
            RefToText::default(),
            Fanout::default(),
            Zip::new(sub_arrow(), sub_arrow()),
        );

        let (_a, _b) = arrow.forward(&mut (), 42);
    }

    #[test]
    fn test_fanout3() {
        let mut f: Fanout3<u32> = Fanout3::default();
        let (a, b, c) = f.forward(&mut (), 7);
        assert_eq!((a, b, c), (7, 7, 7));
    }

    #[test]
    fn test_zip3() {
        let mut z = Zip3::new(
            RefToText::<u32>::default(),
            RefToText::<u32>::default(),
            RefToText::<u32>::default(),
        );
        let (a, b, c) = z.forward(&mut (), (1, 2, 3));
        assert_eq!((&*a, &*b, &*c), ("1", "2", "3"));
    }

    #[test]
    fn test_zipok3() {
        let mut s: ZipOk3<u32, u32, u32, String> = ZipOk3::default();
        let result = s.forward(&mut (), (Ok(1), Ok(2), Ok(3)));
        assert_eq!(result.unwrap(), (1, 2, 3));

        let result_err: Result<(u32, u32, u32), String> =
            s.forward(&mut (), (Ok(1), Err("fail".to_string()), Ok(3)));
        assert!(result_err.is_err());
    }

    #[test]
    fn test_then() {
        let mut t = Then::new(RefToText::<u32>::default(), RefCharsArr::<2>);
        let out = t.forward(&mut (), 42);
        assert!(matches!(out, Ok(['4', '2'])));
    }

    #[test]
    fn test_try_fold_range() {
        let mut fold = TryFoldRange::<AddIndex, usize, &'static str>::new(AddIndex);
        let out = fold.forward(&mut (), (0, 0..5));
        assert!(matches!(out, Ok(10)));
    }

    #[test]
    fn test_try_fold_vec() {
        let mut fold = TryFoldVec::<AddItem, usize, usize, &'static str>::new(AddItem);
        let out = fold.forward(&mut (), (10, vec![1, 2, 3]));
        assert!(matches!(out, Ok(16)));
    }

    #[test]
    fn test_map_vec() {
        let mut map = MapVec::<ToText, usize, String>::new(ToText);
        let out = map.forward(&mut (), vec![3, 4, 5]);
        assert_eq!(out, vec!["3".to_string(), "4".to_string(), "5".to_string()]);
    }

    #[test]
    fn test_try_map_vec() {
        let mut map = TryMapVec::<ToOddFail, usize, usize, &'static str>::new(ToOddFail);
        let out_ok = map.forward(&mut (), vec![2, 4, 6]);
        assert!(matches!(out_ok, Ok(v) if v == vec![2, 4, 6]));

        let out_err = map.forward(&mut (), vec![2, 3, 4]);
        assert!(matches!(out_err, Err("odd")));
    }

    #[test]
    fn test_map_err() {
        let mut map = MapErr::<FailI32, usize, usize, i32, i64>::new(FailI32);
        let out = map.forward(&mut (), 0);
        assert!(matches!(out, Err(7_i64)));
    }

    // -- Switch combinator tests --

    struct EvenOddIndex(usize);
    impl SwitchIndex for EvenOddIndex {
        fn index(&self) -> usize {
            self.0
        }
    }

    struct EvenOddSelector;
    #[primitive(property = Arrow)]
    impl Combinator for EvenOddSelector {
        type In = usize;
        type Out = EvenOddIndex;
        fn forward(&mut self, _ctx: &mut (), input: Self::In) -> Self::Out {
            EvenOddIndex(input % 2)
        }
    }

    struct DoubleIt;
    #[primitive(property = Arrow)]
    impl Combinator for DoubleIt {
        type In = usize;
        type Out = usize;
        fn forward(&mut self, _ctx: &mut (), input: Self::In) -> Self::Out {
            input * 2
        }
    }

    struct TripleIt;
    #[primitive(property = Arrow)]
    impl Combinator for TripleIt {
        type In = usize;
        type Out = usize;
        fn forward(&mut self, _ctx: &mut (), input: Self::In) -> Self::Out {
            input * 3
        }
    }

    struct AddTen;
    #[primitive(property = Arrow)]
    impl Combinator for AddTen {
        type In = usize;
        type Out = usize;
        fn forward(&mut self, _ctx: &mut (), input: Self::In) -> Self::Out {
            input + 10
        }
    }

    struct AddHundred;
    #[primitive(property = Arrow)]
    impl Combinator for AddHundred {
        type In = usize;
        type Out = usize;
        fn forward(&mut self, _ctx: &mut (), input: Self::In) -> Self::Out {
            input + 100
        }
    }

    #[test]
    fn test_switch() {
        let mut sw = Switch::new(EvenOddSelector, DoubleIt, TripleIt);
        // 4 is even (index 0) -> DoubleIt -> 8
        assert_eq!(sw.forward(&mut (), 4), 8);
        // 5 is odd (index 1) -> TripleIt -> 15
        assert_eq!(sw.forward(&mut (), 5), 15);
    }

    fn pick_even_odd(input: &usize) -> usize {
        input % 2
    }

    #[test]
    fn test_switch_ref() {
        let mut sw = SwitchRef::new(pick_even_odd, DoubleIt, TripleIt);
        assert_eq!(sw.forward(&mut (), 4), 8);
        assert_eq!(sw.forward(&mut (), 5), 15);
    }

    struct Mod3Index(usize);
    impl SwitchIndex for Mod3Index {
        fn index(&self) -> usize {
            self.0
        }
    }

    struct Mod3Selector;
    #[primitive(property = Arrow)]
    impl Combinator for Mod3Selector {
        type In = usize;
        type Out = Mod3Index;
        fn forward(&mut self, _ctx: &mut (), input: Self::In) -> Self::Out {
            Mod3Index(input % 3)
        }
    }

    #[test]
    fn test_switch3() {
        let mut sw = Switch3::new(Mod3Selector, DoubleIt, TripleIt, AddTen);
        // 6 % 3 == 0 -> DoubleIt -> 12
        assert_eq!(sw.forward(&mut (), 6), 12);
        // 7 % 3 == 1 -> TripleIt -> 21
        assert_eq!(sw.forward(&mut (), 7), 21);
        // 8 % 3 == 2 -> AddTen -> 18
        assert_eq!(sw.forward(&mut (), 8), 18);
    }

    enum Quad {
        A,
        B,
        C,
        D,
    }
    impl SwitchIndex for Quad {
        fn index(&self) -> usize {
            match self {
                Quad::A => 0,
                Quad::B => 1,
                Quad::C => 2,
                Quad::D => 3,
            }
        }
    }

    struct QuadSelector;
    #[primitive(property = Arrow)]
    impl Combinator for QuadSelector {
        type In = usize;
        type Out = Quad;
        fn forward(&mut self, _ctx: &mut (), input: Self::In) -> Self::Out {
            match input % 4 {
                0 => Quad::A,
                1 => Quad::B,
                2 => Quad::C,
                _ => Quad::D,
            }
        }
    }

    #[test]
    fn test_switch4() {
        let mut sw = Switch4::new(QuadSelector, DoubleIt, TripleIt, AddTen, AddHundred);
        // 8 % 4 == 0 -> DoubleIt -> 16
        assert_eq!(sw.forward(&mut (), 8), 16);
        // 9 % 4 == 1 -> TripleIt -> 27
        assert_eq!(sw.forward(&mut (), 9), 27);
        // 10 % 4 == 2 -> AddTen -> 20
        assert_eq!(sw.forward(&mut (), 10), 20);
        // 11 % 4 == 3 -> AddHundred -> 111
        assert_eq!(sw.forward(&mut (), 11), 111);
    }

    #[test]
    fn test_inject_const() {
        let mut inject = InjectConst::<i32, &str>::new("hello");
        assert_eq!(inject.forward(&mut (), 42), (42, "hello"));
        assert_eq!(inject.forward(&mut (), 7), (7, "hello"));
    }

    #[test]
    fn test_vec_len() {
        let mut vec_len = VecLen::<u32>::default();
        assert_eq!(vec_len.forward(&mut (), vec![1, 2, 3, 4]), 4);
    }

    #[test]
    fn test_eq() {
        let mut eq = Eq::<usize>::default();
        assert!(eq.forward(&mut (), (7, 7)));
        assert!(!eq.forward(&mut (), (7, 8)));
    }

    #[test]
    fn test_inject_const_in_pipeline() {
        // InjectConst + a combinator that uses both values
        let mut flow = Then::new(InjectConst::<usize, usize>::new(100), AddIndex);
        // (5, 100) -> AddIndex -> Ok(105)
        assert_eq!(Combinator::forward(&mut flow, &mut (), 5), Ok(105));
    }
}
