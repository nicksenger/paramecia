macro_rules! forward_1 {
    ($op:literal, $s:ty) => {{
        tracing::trace_span!(
            $op,
            input_shape = std::any::type_name::<<$s as glowstick::ShapeDiagnostic>::Out>()
        )
        .entered()
    }};
}
pub(crate) use forward_1;

macro_rules! forward_2 {
    ($op:literal, $s1:ty, $s2:ty) => {{
        tracing::trace_span!(
            $op,
            input_shape_0 = std::any::type_name::<<$s1 as glowstick::ShapeDiagnostic>::Out>(),
            input_shape_1 = std::any::type_name::<<$s2 as glowstick::ShapeDiagnostic>::Out>()
        )
        .entered()
    }};
}
pub(crate) use forward_2;

macro_rules! forward_3 {
    ($op:literal, $s1:ty, $s2:ty, $s3:ty) => {{
        tracing::trace_span!(
            $op,
            input_shape_0 = std::any::type_name::<<$s1 as glowstick::ShapeDiagnostic>::Out>(),
            input_shape_1 = std::any::type_name::<<$s2 as glowstick::ShapeDiagnostic>::Out>(),
            input_shape_2 = std::any::type_name::<<$s3 as glowstick::ShapeDiagnostic>::Out>()
        )
        .entered()
    }};
}
pub(crate) use forward_3;

macro_rules! forward_vec {
    ($op:literal, $s:ty) => {{
        tracing::trace_span!(
            $op,
            input_shape = std::any::type_name::<<$s as glowstick::ShapeDiagnostic>::Out>()
        )
        .entered()
    }};
}
pub(crate) use forward_vec;

macro_rules! forward_output {
    ($op:literal, $s:ty) => {{
        tracing::trace_span!(
            $op,
            output_shape = std::any::type_name::<<$s as glowstick::ShapeDiagnostic>::Out>()
        )
        .entered()
    }};
}
pub(crate) use forward_output;
