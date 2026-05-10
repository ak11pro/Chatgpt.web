"use client";

import * as THREE from "three";
import { Suspense, useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { Html, Float, Sparkles, Trail, Text, useTexture } from "@react-three/drei";
import { EffectComposer, Bloom, Vignette, Noise } from "@react-three/postprocessing";
import { motion, AnimatePresence } from "framer-motion";
import gsap from "gsap";
import Lenis from "lenis";

const PLANETS = [
  {
    id: "01",
    title: "Planning & Idea Validation",
    body: "I understand the client’s idea, target audience, and goals before starting the website.",
    color: "#6b7cff",
    emissive: "#7085ff",
    position: [-4.5, 0.5, -2],
  },
  {
    id: "02",
    title: "UI/UX Design Approach",
    body: "I create clean, modern, and user-friendly designs based on global trends with perfect colors, fonts, spacing, and full responsiveness.",
    color: "#b34dff",
    emissive: "#f04dff",
    position: [0.5, -1.3, -1],
  },
  {
    id: "03",
    title: "Technology Stack Used",
    body: "I build websites using HTML, CSS, JavaScript, React, and modern technologies.",
    color: "#00aaff",
    emissive: "#00ffd5",
    position: [5, 1.1, -3],
  },
  {
    id: "04",
    title: "Responsive Design Implementation",
    body: "My websites work perfectly on mobile, tablet, and desktop devices.",
    color: "#7b8bff",
    emissive: "#c8dbff",
    position: [-1.8, 2.3, -1.5],
  },
  {
    id: "05",
    title: "Performance & Optimization",
    body: "I create fast, smooth, and highly interactive websites using clean code and optimized assets.",
    color: "#6f6fff",
    emissive: "#18deff",
    position: [3.5, -2.1, -2.2],
  },
  {
    id: "06",
    title: "Features & Functionality",
    body: "I add all required features including forms, animations, APIs, dashboards, and user interactions.",
    color: "#8b5cf6",
    emissive: "#9d6dff",
    position: [-5.7, -2.6, -4.1],
  },
  {
    id: "07",
    title: "Testing & Final Delivery",
    body: "I test for bugs, speed, and responsiveness to deliver a complete, ready-to-use product.",
    color: "#6ba4ff",
    emissive: "#ffffff",
    position: [6.5, 2.9, -5],
  },
];

function useSmoothScroll() {
  useEffect(() => {
    const lenis = new Lenis({ lerp: 0.08, smoothWheel: true });
    let frame = 0;
    const raf = (t) => {
      lenis.raf(t);
      frame = requestAnimationFrame(raf);
    };
    frame = requestAnimationFrame(raf);
    return () => {
      cancelAnimationFrame(frame);
      lenis.destroy();
    };
  }, []);
}

function Starfield() {
  const points = useRef();
  const positions = useMemo(() => {
    const count = 5000;
    const arr = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      arr[i * 3] = (Math.random() - 0.5) * 120;
      arr[i * 3 + 1] = (Math.random() - 0.5) * 90;
      arr[i * 3 + 2] = -Math.random() * 120;
    }
    return arr;
  }, []);

  useFrame((_, delta) => {
    if (!points.current) return;
    points.current.rotation.y += delta * 0.01;
    points.current.rotation.x += delta * 0.003;
  });

  return (
    <points ref={points}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" count={positions.length / 3} array={positions} itemSize={3} />
      </bufferGeometry>
      <pointsMaterial color="#a9b7ff" size={0.06} sizeAttenuation transparent opacity={0.9} depthWrite={false} />
    </points>
  );
}

function Planet({ data, isActive, onOpen }) {
  const mesh = useRef();
  const ring = useRef();
  useFrame((state, delta) => {
    if (!mesh.current || !ring.current) return;
    mesh.current.rotation.y += delta * (isActive ? 0.6 : 0.22);
    mesh.current.position.y += Math.sin(state.clock.elapsedTime + data.position[0]) * 0.002;
    ring.current.rotation.z += delta * 0.3;
  });

  return (
    <group position={data.position}>
      <mesh ref={mesh} onClick={() => onOpen(data)}>
        <sphereGeometry args={[1.1, 64, 64]} />
        <meshStandardMaterial color={data.color} emissive={data.emissive} emissiveIntensity={isActive ? 1.9 : 1.15} roughness={0.38} metalness={0.25} />
      </mesh>
      <mesh ref={ring} rotation={[Math.PI / 2.8, 0, 0]}>
        <torusGeometry args={[1.6, 0.03, 16, 100]} />
        <meshBasicMaterial color="#d8cfff" transparent opacity={0.7} />
      </mesh>
      <Sparkles count={isActive ? 90 : 40} size={2.2} speed={0.2} scale={[3, 3, 3]} color={data.emissive} />
      <Html distanceFactor={8} position={[0, -1.9, 0]}>
        <button
          type="button"
          className="rounded-full border border-fuchsia-400/60 bg-black/40 px-3 py-1 text-xs text-fuchsia-100 backdrop-blur-sm transition hover:scale-105"
          onClick={() => onOpen(data)}
        >
          Explore {data.id}
        </button>
      </Html>
    </group>
  );
}

function CameraRig({ target }) {
  const { camera } = useThree();
  useFrame(() => {
    const p = target?.position || [0, 0, 0];
    camera.position.x += (p[0] * 0.5 - camera.position.x) * 0.03;
    camera.position.y += (p[1] * 0.5 - camera.position.y) * 0.03;
    camera.position.z += ((target ? 5 : 9) - camera.position.z) * 0.04;
    camera.lookAt(p[0], p[1], p[2]);
  });
  return null;
}

function SpaceScene({ activePlanet, setActivePlanet }) {
  return (
    <Canvas dpr={[1, 1.7]} camera={{ position: [0, 0, 9], fov: 52 }} gl={{ antialias: true, powerPreference: "high-performance" }}>
      <color attach="background" args={["#03020d"]} />
      <fog attach="fog" args={["#070619", 10, 45]} />
      <ambientLight intensity={0.55} />
      <pointLight position={[0, 0, 10]} intensity={3} color="#8dabff" />
      <Suspense fallback={null}>
        <Starfield />
        <Sparkles count={220} size={2.6} speed={0.15} scale={[40, 26, 24]} color="#8dc9ff" />
        {PLANETS.map((planet) => (
          <Float key={planet.id} speed={1.2} rotationIntensity={0.22} floatIntensity={0.45}>
            <Planet data={planet} isActive={activePlanet?.id === planet.id} onOpen={setActivePlanet} />
          </Float>
        ))}
        <CameraRig target={activePlanet} />
      </Suspense>
      <EffectComposer>
        <Bloom luminanceThreshold={0.1} luminanceSmoothing={0.8} intensity={1.45} />
        <Vignette eskil={false} offset={0.23} darkness={0.95} />
        <Noise opacity={0.02} />
      </EffectComposer>
    </Canvas>
  );
}

export default function FuturisticSpacePortfolio() {
  const [loaded, setLoaded] = useState(false);
  const [entered, setEntered] = useState(false);
  const [activePlanet, setActivePlanet] = useState(null);
  const cursor = useRef(null);
  const comet = useRef(null);

  useSmoothScroll();

  useEffect(() => {
    const timer = setTimeout(() => setLoaded(true), 2400);
    return () => clearTimeout(timer);
  }, []);

  useEffect(() => {
    const move = (e) => {
      gsap.to(cursor.current, { x: e.clientX, y: e.clientY, duration: 0.35, ease: "power3.out" });
      gsap.to(comet.current, { x: e.clientX, y: e.clientY, duration: 0.75, ease: "power3.out" });
    };
    window.addEventListener("mousemove", move);
    return () => window.removeEventListener("mousemove", move);
  }, []);

  return (
    <main className="relative min-h-screen overflow-hidden bg-[#03020d] text-white">
      <div ref={cursor} className="pointer-events-none fixed left-0 top-0 z-[99] h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-full bg-cyan-300 shadow-[0_0_40px_10px_rgba(34,211,238,0.8)]" />
      <div ref={comet} className="pointer-events-none fixed left-0 top-0 z-[98] h-12 w-12 -translate-x-1/2 -translate-y-1/2 rounded-full bg-fuchsia-400/20 blur-xl" />

      <AnimatePresence>
        {!loaded && (
          <motion.section initial={{ opacity: 1 }} exit={{ opacity: 0 }} className="fixed inset-0 z-50 flex items-center justify-center bg-[#040212]">
            <motion.div animate={{ scale: [0.9, 1.15, 1], rotate: [0, 180, 360] }} transition={{ duration: 2.1, repeat: Infinity, ease: "easeInOut" }} className="h-20 w-20 rounded-full border border-cyan-300/60 shadow-[0_0_80px_#22d3ee]" />
            <p className="absolute bottom-24 text-xs uppercase tracking-[0.35em] text-cyan-100/70">Calibrating Galactic Interface...</p>
          </motion.section>
        )}
      </AnimatePresence>

      <section className="relative h-screen">
        <SpaceScene activePlanet={activePlanet} setActivePlanet={setActivePlanet} />
        {!entered && (
          <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
            <motion.div initial={{ y: 90, opacity: 0 }} animate={{ y: 0, opacity: 1 }} className="pointer-events-auto rounded-3xl border border-cyan-300/40 bg-black/45 p-8 text-center backdrop-blur-xl">
              <h1 className="mb-3 text-2xl font-semibold tracking-wide">Cinematic Space Portfolio</h1>
              <p className="mb-5 text-sm text-cyan-100/80">Enter the universe and explore each planet as a portfolio chapter.</p>
              <button
                type="button"
                onClick={() => setEntered(true)}
                className="rounded-full border border-cyan-300/80 px-7 py-3 text-sm uppercase tracking-[0.25em] text-cyan-100 shadow-[0_0_32px_rgba(34,211,238,0.45)] transition hover:scale-105"
              >
                Enter
              </button>
            </motion.div>
          </div>
        )}
      </section>

      <AnimatePresence>
        {entered && activePlanet && (
          <motion.aside
            initial={{ x: 450, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: 500, opacity: 0 }}
            className="fixed right-6 top-1/2 z-40 w-[min(92vw,410px)] -translate-y-1/2 rounded-3xl border border-white/20 bg-white/10 p-6 backdrop-blur-2xl"
          >
            <p className="mb-2 text-xs tracking-[0.3em] text-fuchsia-200">PLANET {activePlanet.id}</p>
            <h2 className="mb-3 text-2xl font-semibold">{activePlanet.title}</h2>
            <p className="text-sm leading-relaxed text-cyan-100/80">{activePlanet.body}</p>
            <button type="button" onClick={() => setActivePlanet(null)} className="mt-6 rounded-full border border-fuchsia-300/60 px-4 py-2 text-xs uppercase tracking-[0.2em]">
              Close
            </button>
          </motion.aside>
        )}
      </AnimatePresence>
    </main>
  );
}
