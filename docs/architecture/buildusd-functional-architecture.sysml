package BuildUSDFunctionalArchitecture {
    doc /*
     * Functional architecture of BuildUSD, derived from repository behavior.
     *
     * Modelling approach:
     * - The primary objects are state-bearing information artifacts.
     * - Functions are verb-object actions that transition those artifacts.
     * - Main and auxiliary information-flow chains are aggregated explicitly.
     * - Logical implementation parts perform the modeled behavior.
     *
     * The model uses the OMG SysML 2.0 textual notation.
     */

    package StateAndFlowTypes {
        attribute def URI;
        attribute def Name;
        attribute def Status;
        attribute def Count;
        attribute def UnitScale;
        attribute def CoordinateSystem;
        attribute def CoordinateVector;
        attribute def Format;
        attribute def Mode;
        attribute def Flag;
        attribute def CacheKey;

        item def ConversionRequest {
            attribute inputURI : URI;
            attribute outputURI : URI;
            attribute format : Format;
            attribute anchorMode : Mode;
            attribute geospatialMode : Mode;
            attribute offline : Flag;
            attribute checkpoint : Flag;
        }

        item def FederationRequest {
            attribute outputURI : URI;
            attribute referenceFrame : Mode;
            attribute anchorMode : Mode;
            attribute missingAnchorPolicy : Mode;
            attribute rebuild : Flag;
        }

        item def EnrichmentJob {
            attribute jobId : Name;
            attribute sourceURI : URI;
            attribute outputURI : URI;
            attribute detailEngine : Mode;
            attribute cacheKey : CacheKey;
            attribute status : Status;
        }

        item def ConfigurationState {
            attribute manifestURI : URI;
            attribute ruleCount : Count;
            attribute projectedCRS : CoordinateSystem;
            attribute geodeticCRS : CoordinateSystem;
            attribute basePoint : CoordinateVector;
            attribute optionCount : Count;
        }

        item def CancellationState {
            attribute cancelled : Flag;
        }

        item def SourceSetState {
            attribute rootURI : URI;
            attribute candidateCount : Count;
            attribute accessible : Flag;
        }

        item def SelectedSourceSetState {
            attribute selectedCount : Count;
            attribute excludedCount : Count;
            attribute remoteCount : Count;
        }

        item def ParsedModelSetState {
            attribute modelCount : Count;
            attribute schema : Name;
            attribute unitScaleToMeters : UnitScale;
            attribute elementCount : Count;
            attribute coordinateOperationPresent : Flag;
        }

        item def ResolvedModelSetState {
            attribute projectedCRS : CoordinateSystem;
            attribute geodeticCRS : CoordinateSystem;
            attribute strategy : Mode;
            attribute modelOffset : CoordinateVector;
            attribute rotationApplied : Flag;
            attribute localizationApplied : Flag;
        }

        item def PrototypeCacheSetState {
            attribute typePrototypeCount : Count;
            attribute hashPrototypeCount : Count;
            attribute instanceCount : Count;
            attribute annotationCount : Count;
            attribute semanticDetailCount : Count;
        }

        item def LayerSetState {
            attribute format : Format;
            attribute prototypeLayerURI : URI;
            attribute materialLayerURI : URI;
            attribute instanceLayerURI : URI;
            attribute geometry2DLayerURI : URI;
            attribute metersPerUnit : UnitScale;
        }

        item def AnchoredStageSetState {
            attribute stageCount : Count;
            attribute anchorMode : Mode;
            attribute projectedCRS : CoordinateSystem;
            attribute geospatialMetadataPresent : Flag;
            attribute saved : Flag;
        }

        item def SemanticGraphSetState {
            attribute graphCount : Count;
            attribute identityLinkCount : Count;
            attribute unresolvedDetailCount : Count;
        }

        item def ConversionResultSetState {
            attribute convertedCount : Count;
            attribute failedCount : Count;
            attribute totalElementCount : Count;
            attribute status : Status;
        }

        item def RoutedStageSetState {
            attribute payloadCount : Count;
            attribute masterGroupCount : Count;
            attribute skippedMasterCount : Count;
        }

        item def InspectedPayloadSetState {
            attribute payloadCount : Count;
            attribute anchoredCount : Count;
            attribute missingAnchorCount : Count;
            attribute sourceCRS : CoordinateSystem;
        }

        item def AlignedPayloadSetState {
            attribute payloadCount : Count;
            attribute targetCRS : CoordinateSystem;
            attribute referenceFrame : Mode;
            attribute federationOrigin : CoordinateVector;
        }

        item def SiteMasterSetState {
            attribute masterCount : Count;
            attribute payloadCount : Count;
            attribute idempotent : Flag;
        }

        item def FederatedProjectState {
            attribute projectStageURI : URI;
            attribute siteMasterCount : Count;
            attribute referenceFrame : Mode;
            attribute validated : Flag;
        }

        item def ClaimedJobState {
            attribute jobId : Name;
            attribute runningURI : URI;
            attribute cacheKey : CacheKey;
            attribute status : Status;
        }

        item def DetailPlanState {
            attribute jobId : Name;
            attribute targetCount : Count;
            attribute objectScope : Flag;
            attribute cacheHit : Flag;
        }

        item def DetailArtifactSetState {
            attribute stageCount : Count;
            attribute semanticGraphCount : Count;
            attribute replacementManifestURI : URI;
            attribute cacheHit : Flag;
        }

        item def JobResultState {
            attribute jobId : Name;
            attribute resultURI : URI;
            attribute archiveURI : URI;
            attribute status : Status;
        }

        item def RuntimeContextState {
            attribute mode : Mode;
            attribute pxrAvailable : Flag;
            attribute nucleusAvailable : Flag;
        }
    }

    package FunctionalBehavior {
        private import StateAndFlowTypes::*;

        action def 'Interpret Conversion Intent' {
            in request : ConversionRequest;
            in configuration : ConfigurationState;
            out interpretedRequest : ConversionRequest;
        }

        action def 'Discover IFC Sources' {
            in request : ConversionRequest;
            in sources : SourceSetState;
            out selectedSources : SelectedSourceSetState;
        }

        action def 'Prepare USD Runtime' {
            in request : ConversionRequest;
            out runtime : RuntimeContextState;
        }

        action def 'Load IFC Models' {
            in selectedSources : SelectedSourceSetState;
            in runtime : RuntimeContextState;
            out parsedModels : ParsedModelSetState;
        }

        action def 'Resolve Model Georeference' {
            in parsedModels : ParsedModelSetState;
            in configuration : ConfigurationState;
            out resolvedModels : ResolvedModelSetState;
        }

        action def 'Build Geometry Prototypes' {
            in resolvedModels : ResolvedModelSetState;
            out prototypeCaches : PrototypeCacheSetState;
        }

        action def 'Author Prototype Layers' {
            in prototypeCaches : PrototypeCacheSetState;
            out prototypeLayers : LayerSetState;
        }

        action def 'Author Material Layers' {
            in prototypeCaches : PrototypeCacheSetState;
            in prototypeLayers : LayerSetState;
            out materialLayers : LayerSetState;
        }

        action def 'Author Instance Layers' {
            in prototypeCaches : PrototypeCacheSetState;
            in materialLayers : LayerSetState;
            out instanceLayers : LayerSetState;
        }

        action def 'Author 2D Geometry Layers' {
            in prototypeCaches : PrototypeCacheSetState;
            in instanceLayers : LayerSetState;
            out authoredLayers : LayerSetState;
        }

        action def 'Anchor USD Stages' {
            in authoredLayers : LayerSetState;
            in resolvedModels : ResolvedModelSetState;
            out anchoredStages : AnchoredStageSetState;
        }

        action def 'Persist Semantic Graphs' {
            in prototypeCaches : PrototypeCacheSetState;
            in anchoredStages : AnchoredStageSetState;
            out semanticGraphs : SemanticGraphSetState;
        }

        action def 'Finalize Conversion Results' {
            in anchoredStages : AnchoredStageSetState;
            in semanticGraphs : SemanticGraphSetState;
            out results : ConversionResultSetState;
        }

        action def <ConvertIfcAssetsDef> 'Convert IFC Assets' {
            in item request : ConversionRequest;
            in item sources : SourceSetState;
            in item configuration : ConfigurationState;
            in item cancellation : CancellationState;
            out item stages : AnchoredStageSetState;
            out item semanticGraphs : SemanticGraphSetState;
            out item results : ConversionResultSetState;

            bind interpret.request = request;
            bind interpret.configuration = configuration;
            action interpret : 'Interpret Conversion Intent' {
                in request;
                in configuration;
                out interpretedRequest;
            }

            succession flow from interpret.interpretedRequest to discover.request;
            bind discover.sources = sources;
            action discover : 'Discover IFC Sources' {
                in request;
                in sources;
                out selectedSources;
            }

            flow from interpret.interpretedRequest to prepareRuntime.request;
            action prepareRuntime : 'Prepare USD Runtime' {
                in request;
                out runtime;
            }

            succession flow from discover.selectedSources to loadModels.selectedSources;
            flow from prepareRuntime.runtime to loadModels.runtime;
            action loadModels : 'Load IFC Models' {
                in selectedSources;
                in runtime;
                out parsedModels;
            }

            succession flow from loadModels.parsedModels to resolveGeoreference.parsedModels;
            bind resolveGeoreference.configuration = configuration;
            action resolveGeoreference : 'Resolve Model Georeference' {
                in parsedModels;
                in configuration;
                out resolvedModels;
            }

            succession flow from resolveGeoreference.resolvedModels to buildPrototypes.resolvedModels;
            action buildPrototypes : 'Build Geometry Prototypes' {
                in resolvedModels;
                out prototypeCaches;
            }

            succession flow from buildPrototypes.prototypeCaches to authorPrototypes.prototypeCaches;
            action authorPrototypes : 'Author Prototype Layers' {
                in prototypeCaches;
                out prototypeLayers;
            }

            flow from buildPrototypes.prototypeCaches to authorMaterials.prototypeCaches;
            succession flow from authorPrototypes.prototypeLayers to authorMaterials.prototypeLayers;
            action authorMaterials : 'Author Material Layers' {
                in prototypeCaches;
                in prototypeLayers;
                out materialLayers;
            }

            flow from buildPrototypes.prototypeCaches to authorInstances.prototypeCaches;
            succession flow from authorMaterials.materialLayers to authorInstances.materialLayers;
            action authorInstances : 'Author Instance Layers' {
                in prototypeCaches;
                in materialLayers;
                out instanceLayers;
            }

            flow from buildPrototypes.prototypeCaches to author2D.prototypeCaches;
            succession flow from authorInstances.instanceLayers to author2D.instanceLayers;
            action author2D : 'Author 2D Geometry Layers' {
                in prototypeCaches;
                in instanceLayers;
                out authoredLayers;
            }

            succession flow from author2D.authoredLayers to anchorStages.authoredLayers;
            flow from resolveGeoreference.resolvedModels to anchorStages.resolvedModels;
            action anchorStages : 'Anchor USD Stages' {
                in authoredLayers;
                in resolvedModels;
                out anchoredStages;
            }

            flow from buildPrototypes.prototypeCaches to persistSemantics.prototypeCaches;
            succession flow from anchorStages.anchoredStages to persistSemantics.anchoredStages;
            action persistSemantics : 'Persist Semantic Graphs' {
                in prototypeCaches;
                in anchoredStages;
                out semanticGraphs;
            }

            flow from anchorStages.anchoredStages to finalizeResults.anchoredStages;
            succession flow from persistSemantics.semanticGraphs to finalizeResults.semanticGraphs;
            action finalizeResults : 'Finalize Conversion Results' {
                in anchoredStages;
                in semanticGraphs;
                out results;
            }

            bind anchorStages.anchoredStages = stages;
            bind persistSemantics.semanticGraphs = semanticGraphs;
            bind finalizeResults.results = results;
        }

        action def 'Plan Stage Federation' {
            in request : FederationRequest;
            in stages : AnchoredStageSetState;
            in configuration : ConfigurationState;
            out routedStages : RoutedStageSetState;
        }

        action def 'Inspect Payload Anchors' {
            in routedStages : RoutedStageSetState;
            out inspectedPayloads : InspectedPayloadSetState;
        }

        action def 'Align Stage Payloads' {
            in request : FederationRequest;
            in configuration : ConfigurationState;
            in inspectedPayloads : InspectedPayloadSetState;
            out alignedPayloads : AlignedPayloadSetState;
        }

        action def 'Compose Site Masters' {
            in alignedPayloads : AlignedPayloadSetState;
            out siteMasters : SiteMasterSetState;
        }

        action def 'Compose Overall Master' {
            in siteMasters : SiteMasterSetState;
            in configuration : ConfigurationState;
            out project : FederatedProjectState;
        }

        action def 'Validate Federated Project' {
            in project : FederatedProjectState;
            out validatedProject : FederatedProjectState;
        }

        action def <FederateUsdStagesDef> 'Federate USD Stages' {
            in item request : FederationRequest;
            in item stages : AnchoredStageSetState;
            in item configuration : ConfigurationState;
            out item project : FederatedProjectState;

            bind plan.request = request;
            bind plan.stages = stages;
            bind plan.configuration = configuration;
            action plan : 'Plan Stage Federation' {
                in request;
                in stages;
                in configuration;
                out routedStages;
            }

            succession flow from plan.routedStages to inspect.inspectedPayloads;
            action inspect : 'Inspect Payload Anchors' {
                in routedStages;
                out inspectedPayloads;
            }

            bind align.request = request;
            bind align.configuration = configuration;
            succession flow from inspect.inspectedPayloads to align.inspectedPayloads;
            action align : 'Align Stage Payloads' {
                in request;
                in configuration;
                in inspectedPayloads;
                out alignedPayloads;
            }

            succession flow from align.alignedPayloads to composeSites.alignedPayloads;
            action composeSites : 'Compose Site Masters' {
                in alignedPayloads;
                out siteMasters;
            }

            succession flow from composeSites.siteMasters to composeOverall.siteMasters;
            bind composeOverall.configuration = configuration;
            action composeOverall : 'Compose Overall Master' {
                in siteMasters;
                in configuration;
                out project;
            }

            succession flow from composeOverall.project to validateProject.project;
            action validateProject : 'Validate Federated Project' {
                in project;
                out validatedProject;
            }

            bind validateProject.validatedProject = project;
        }

        action def 'Claim Enrichment Job' {
            in job : EnrichmentJob;
            out claimedJob : ClaimedJobState;
        }

        action def 'Resolve Detail Targets' {
            in claimedJob : ClaimedJobState;
            in semanticGraphs : SemanticGraphSetState;
            out detailPlan : DetailPlanState;
        }

        action def 'Satisfy Targeted Detail' {
            in detailPlan : DetailPlanState;
            in configuration : ConfigurationState;
            out detailArtifacts : DetailArtifactSetState;
        }

        action def 'Publish Replacement Manifest' {
            in detailArtifacts : DetailArtifactSetState;
            out publishedArtifacts : DetailArtifactSetState;
        }

        action def 'Archive Job Outcome' {
            in publishedArtifacts : DetailArtifactSetState;
            in claimedJob : ClaimedJobState;
            out result : JobResultState;
        }

        action def <RunTargetedEnrichmentJobDef> 'Run Targeted Enrichment Job' {
            in item job : EnrichmentJob;
            in item semanticGraphs : SemanticGraphSetState;
            in item configuration : ConfigurationState;
            out item result : JobResultState;

            bind claim.job = job;
            action claim : 'Claim Enrichment Job' {
                in job;
                out claimedJob;
            }

            succession flow from claim.claimedJob to resolveTargets.claimedJob;
            bind resolveTargets.semanticGraphs = semanticGraphs;
            action resolveTargets : 'Resolve Detail Targets' {
                in claimedJob;
                in semanticGraphs;
                out detailPlan;
            }

            succession flow from resolveTargets.detailPlan to satisfyDetail.detailPlan;
            bind satisfyDetail.configuration = configuration;
            action satisfyDetail : 'Satisfy Targeted Detail' {
                in detailPlan;
                in configuration;
                out detailArtifacts;
            }

            succession flow from satisfyDetail.detailArtifacts to publishManifest.detailArtifacts;
            action publishManifest : 'Publish Replacement Manifest' {
                in detailArtifacts;
                out publishedArtifacts;
            }

            succession flow from publishManifest.publishedArtifacts to archiveOutcome.publishedArtifacts;
            flow from claim.claimedJob to archiveOutcome.claimedJob;
            action archiveOutcome : 'Archive Job Outcome' {
                in publishedArtifacts;
                in claimedJob;
                out result;
            }

            bind archiveOutcome.result = result;
        }

        action convertIFCAssets : 'Convert IFC Assets';
        action federateUSDStages : 'Federate USD Stages';
        action runTargetedEnrichmentJob : 'Run Targeted Enrichment Job';
    }

    package SystemBoundary {
        private import StateAndFlowTypes::*;
        private import FunctionalBehavior::*;

        port def ConversionCommandPort {
            in item request : ConversionRequest;
            out item result : ConversionResultSetState;
        }

        port def FederationCommandPort {
            in item request : FederationRequest;
            out item result : FederatedProjectState;
        }

        port def EnrichmentJobPort {
            in item job : EnrichmentJob;
            out item result : JobResultState;
        }

        port def IFCRepositoryPort {
            in item sources : SourceSetState;
        }

        port def ConfigurationPort {
            in item configuration : ConfigurationState;
        }

        port def USDArtifactPort {
            out item stages : AnchoredStageSetState;
            out item semanticGraphs : SemanticGraphSetState;
            out item project : FederatedProjectState;
        }

        part def BuildUSDSystem {
            port conversionCommand : ConversionCommandPort;
            port federationCommand : FederationCommandPort;
            port enrichmentJob : EnrichmentJobPort;
            port ifcRepository : IFCRepositoryPort;
            port configuration : ConfigurationPort;
            port usdArtifacts : USDArtifactPort;

            perform convertIFCAssets;
            perform federateUSDStages;
            perform runTargetedEnrichmentJob;
        }

        part def OperatorOrAPIClient;
        part def IFCAndConfigurationRepository;
        part def USDOrOmniverseRepository;
        part def DownstreamUSDConsumer;

        part buildUSD : BuildUSDSystem;
        part client : OperatorOrAPIClient;
        part sourceRepository : IFCAndConfigurationRepository;
        part artifactRepository : USDOrOmniverseRepository;
        part consumer : DownstreamUSDConsumer;
    }

    package LogicalImplementation {
        private import FunctionalBehavior::*;

        part def CommandAndAPIAdapter;
        part def ConversionOrchestrator;
        part def ManifestResolver;
        part def SourceIOAdapter;
        part def USDRuntimeAdapter;
        part def GeoreferencePlanner;
        part def IFCGeometryProcessor;
        part def USDAuthoringService;
        part def SemanticService;
        part def FederationOrchestrator;
        part def FederationBuilder;
        part def EnrichmentJobService;
        part def WorkerQueueAdapter;

        part logicalArchitecture {
            part commandAndAPI : CommandAndAPIAdapter;
            part conversionOrchestrator : ConversionOrchestrator;
            part manifestResolver : ManifestResolver;
            part sourceIO : SourceIOAdapter;
            part usdRuntime : USDRuntimeAdapter;
            part georeferencePlanner : GeoreferencePlanner;
            part ifcGeometryProcessor : IFCGeometryProcessor;
            part usdAuthoring : USDAuthoringService;
            part semanticService : SemanticService;
            part federationOrchestrator : FederationOrchestrator;
            part federationBuilder : FederationBuilder;
            part enrichmentJobs : EnrichmentJobService;
            part workerQueue : WorkerQueueAdapter;
        }

        allocation def FunctionToImplementation;

        allocation convertOrchestration allocate
            convertIFCAssets.interpret to logicalArchitecture.conversionOrchestrator;
        allocation sourceDiscovery allocate
            convertIFCAssets.discover to logicalArchitecture.sourceIO;
        allocation runtimePreparation allocate
            convertIFCAssets.prepareRuntime to logicalArchitecture.usdRuntime;
        allocation modelLoading allocate
            convertIFCAssets.loadModels to logicalArchitecture.sourceIO;
        allocation georeferenceResolution allocate
            convertIFCAssets.resolveGeoreference to logicalArchitecture.georeferencePlanner;
        allocation prototypeConstruction allocate
            convertIFCAssets.buildPrototypes to logicalArchitecture.ifcGeometryProcessor;
        allocation prototypeAuthoring allocate
            convertIFCAssets.authorPrototypes to logicalArchitecture.usdAuthoring;
        allocation materialAuthoring allocate
            convertIFCAssets.authorMaterials to logicalArchitecture.usdAuthoring;
        allocation instanceAuthoring allocate
            convertIFCAssets.authorInstances to logicalArchitecture.usdAuthoring;
        allocation geometry2DAuthoring allocate
            convertIFCAssets.author2D to logicalArchitecture.usdAuthoring;
        allocation stageAnchoring allocate
            convertIFCAssets.anchorStages to logicalArchitecture.usdAuthoring;
        allocation semanticPersistence allocate
            convertIFCAssets.persistSemantics to logicalArchitecture.semanticService;
        allocation resultFinalization allocate
            convertIFCAssets.finalizeResults to logicalArchitecture.conversionOrchestrator;

        allocation federationPlanning allocate
            federateUSDStages.plan to logicalArchitecture.federationOrchestrator;
        allocation payloadInspection allocate
            federateUSDStages.inspect to logicalArchitecture.federationBuilder;
        allocation payloadAlignment allocate
            federateUSDStages.align to logicalArchitecture.federationBuilder;
        allocation siteComposition allocate
            federateUSDStages.composeSites to logicalArchitecture.federationBuilder;
        allocation overallComposition allocate
            federateUSDStages.composeOverall to logicalArchitecture.federationOrchestrator;
        allocation federationValidation allocate
            federateUSDStages.validateProject to logicalArchitecture.federationBuilder;

        allocation jobClaiming allocate
            runTargetedEnrichmentJob.claim to logicalArchitecture.workerQueue;
        allocation targetResolution allocate
            runTargetedEnrichmentJob.resolveTargets to logicalArchitecture.semanticService;
        allocation detailSatisfaction allocate
            runTargetedEnrichmentJob.satisfyDetail to logicalArchitecture.enrichmentJobs;
        allocation replacementPublication allocate
            runTargetedEnrichmentJob.publishManifest to logicalArchitecture.enrichmentJobs;
        allocation outcomeArchiving allocate
            runTargetedEnrichmentJob.archiveOutcome to logicalArchitecture.workerQueue;
    }
}
