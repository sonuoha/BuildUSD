def debug_dump_contexts(ifc):
    """Prints out all IfcGeometricRepresentationContext entities in the IFC model, along with their identifiers, types, dimensions, and number of operations."""
    for ctx in ifc.by_type("IfcGeometricRepresentationContext") or []:
        ident = getattr(ctx, "ContextIdentifier", None)
        ctype = getattr(ctx, "ContextType", None)
        dim = getattr(ctx, "CoordinateSpaceDimension", None)
        ops = getattr(ctx, "HasCoordinateOperation", None) or []
        subcontext = getattr(ctx, "HasSubContexts", None) or []
        subctx_identifiers = [
            getattr(sctx, "ContextIdentifier", None) for sctx in subcontext
        ]
        true_north = getattr(ctx, "TrueNorth", None)
        print(
            f"  • GRC id={ctx.id():<6} type={ctype} ident={ident} dim={dim} "
            f"ops={len(ops)} subctx={subctx_identifiers} tn={true_north}"
        )
